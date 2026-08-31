#include "src/core/Loss.hpp"
#include "src/data/Dataset.hpp"
#include "src/model/ModelFactory.hpp"
#include "src/training/Trainer.hpp"
#include "src/training/TrainingDiagnostics.hpp"
#include "src/tuning/CrossValidator.hpp"
#include "src/tuning/SearchSpace.hpp"
#include "src/tuning/TrialConfig.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mpi.h>
#include <numbers>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr double kValidationFraction = 0.10;
constexpr double kOverfitRatio = 0.15;
constexpr double kAoAThresholdRadians =
    10.0 * std::numbers::pi / 180.0;

struct CommandLineOptions {
    bool cross_validate = false;
    bool show_help = false;
    bool diagnostics = true;
    bool verbose_final = true;
    std::string activation = "leakyrelu";
    float leaky_alpha = 0.05f;
    float dropout = 0.0f;
    float learning_rate = 1e-3f;
    float physics_weight = 0.10f;
    float l1_weight = 0.0f;
    float l2_weight = 0.0f;
    float gradient_clip = 1.0f;
    size_t epochs = 200;
    // The canonical 1542-sample holdout split divides exactly into 6 batches
    // of 257, avoiding the six-sample tail that motivated this setting.
    size_t global_batch_size = 257;
    size_t folds = 5;
    size_t validation_interval = 0;
    size_t histogram_bins = 64;
    uint64_t seed = 42;
    std::string train_path = "dataset/cnn_dataset_train.npz";
    std::string test_path = "dataset/cnn_dataset_test.npz";
    std::string results_dir = "results";
    std::string experiment_name = "cnn";
    std::string run_name = "run";
};

struct DatasetSplit {
    std::vector<size_t> training;
    std::vector<size_t> validation;
};

struct BatchPartition {
    size_t offset = 0;
    size_t count = 0;
};

struct TestMetrics {
    double overall_physical_mse = std::numeric_limits<double>::quiet_NaN();
    double low_angle_physical_mse = std::numeric_limits<double>::quiet_NaN();
    double high_angle_physical_mse = std::numeric_limits<double>::quiet_NaN();
    size_t overall_samples = 0;
    size_t low_angle_samples = 0;
    size_t high_angle_samples = 0;
};

struct FinalMetrics {
    size_t selected_epoch = 0;
    double training_physical_mse = std::numeric_limits<double>::quiet_NaN();
    double validation_physical_mse = std::numeric_limits<double>::quiet_NaN();
    double test_physical_mse = std::numeric_limits<double>::quiet_NaN();
};

struct TrainedRun {
    std::unique_ptr<CNNModel> model;
    TrainingResult result;
};

DatasetSplit make_training_validation_split(const Dataset& dataset,
                                            uint64_t seed) {
    std::vector<size_t> shuffled = dataset.all_indices();
    if (shuffled.size() < 2) {
        throw std::invalid_argument(
            "Training dataset must contain at least two samples for validation.");
    }

    std::mt19937_64 generator(seed);
    std::shuffle(shuffled.begin(), shuffled.end(), generator);
    const size_t validation_count = std::max(
        size_t{1}, static_cast<size_t>(shuffled.size() * kValidationFraction));
    if (validation_count >= shuffled.size()) {
        throw std::invalid_argument(
            "Validation split would leave no training samples.");
    }

    return DatasetSplit{
        std::vector<size_t>(shuffled.begin() + validation_count,
                            shuffled.end()),
        std::vector<size_t>(shuffled.begin(),
                            shuffled.begin() + validation_count)};
}

BatchPartition partition_batch(size_t batch_size, int rank, int world_size) {
    const size_t processes = static_cast<size_t>(world_size);
    const size_t base = batch_size / processes;
    const size_t remainder = batch_size % processes;
    const size_t rank_index = static_cast<size_t>(rank);
    return BatchPartition{
        rank_index * base + std::min(rank_index, remainder),
        base + (rank_index < remainder ? 1 : 0)};
}

TestMetrics evaluate_test_metrics(CNNModel& model,
                                  const Dataset& dataset,
                                  std::span<const size_t> indices,
                                  const NormalizationStats& normalization,
                                  size_t global_batch_size,
                                  MPI_Comm communicator) {
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &world_size);

    double local_sums[3]{0.0, 0.0, 0.0};
    unsigned long long local_counts[3]{0, 0, 0};
    std::string local_error;
    try {
        if (global_batch_size == 0) {
            throw std::invalid_argument("Test evaluation batch size must be positive.");
        }
        for (size_t batch_offset = 0;
             batch_offset < indices.size();
             batch_offset += global_batch_size) {
            const size_t global_count = std::min(global_batch_size,
                                                 indices.size() - batch_offset);
            const BatchPartition local =
                partition_batch(global_count, rank, world_size);
            if (local.count == 0) {
                continue;
            }

            const std::span<const size_t> local_indices(
                indices.data() + batch_offset + local.offset, local.count);
            const DataBatch batch = dataset.make_batch(local_indices, normalization);
            const auto predictions = model.predict(batch.sdf, batch.scalars);
            const float* prediction_values = predictions->get_data();
            const float* target_values = batch.targets->get_data();
            const float* alpha_values = batch.alpha_radians->get_data();

            const float batch_physical_mse = Loss::physical_mse(
                predictions, batch.targets,
                static_cast<float>(normalization.target_std));
            local_sums[0] += static_cast<double>(batch_physical_mse) * local.count;
            local_counts[0] += static_cast<unsigned long long>(local.count);

            for (size_t sample = 0; sample < local.count; ++sample) {
                const double error =
                    (static_cast<double>(prediction_values[sample]) -
                     static_cast<double>(target_values[sample])) *
                    static_cast<double>(static_cast<float>(normalization.target_std));
                const double squared_error = error * error;

                const bool high_angle =
                    std::abs(static_cast<double>(alpha_values[sample])) >
                    kAoAThresholdRadians;
                const size_t bucket = high_angle ? 2 : 1;
                local_sums[bucket] += squared_error;
                ++local_counts[bucket];
            }
        }
    } catch (const std::exception& error) {
        local_error = error.what();
    } catch (...) {
        local_error = "unknown local test metric evaluation error";
    }

    int local_failed = local_error.empty() ? 0 : 1;
    int any_failed = local_failed;
    MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX,
                  communicator);
    if (any_failed != 0) {
        if (!local_error.empty()) {
            throw std::runtime_error("Test metric evaluation: " + local_error);
        }
        throw std::runtime_error(
            "Test metric evaluation failed on another MPI rank.");
    }

    double global_sums[3]{0.0, 0.0, 0.0};
    unsigned long long global_counts[3]{0, 0, 0};
    MPI_Allreduce(local_sums, global_sums, 3, MPI_DOUBLE, MPI_SUM,
                  communicator);
    MPI_Allreduce(local_counts, global_counts, 3, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, communicator);

    const auto mean = [&](size_t bucket) {
        return global_counts[bucket] == 0
            ? std::numeric_limits<double>::quiet_NaN()
            : global_sums[bucket] / static_cast<double>(global_counts[bucket]);
    };
    return TestMetrics{
        mean(0),
        mean(1),
        mean(2),
        static_cast<size_t>(global_counts[0]),
        static_cast<size_t>(global_counts[1]),
        static_cast<size_t>(global_counts[2])};
}

void write_test_metrics(const std::filesystem::path& output_directory,
                        const TestMetrics& metrics) {
    std::filesystem::create_directories(output_directory);
    const auto path = output_directory / "test_metrics.csv";
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Failed to write test metrics: " + path.string());
    }

    output << "subset,samples,physical_mse\n" << std::setprecision(17)
           << "overall," << metrics.overall_samples << ','
           << metrics.overall_physical_mse << '\n'
           << "abs_aoa_le_10_deg," << metrics.low_angle_samples << ','
           << metrics.low_angle_physical_mse << '\n'
           << "abs_aoa_gt_10_deg," << metrics.high_angle_samples << ','
           << metrics.high_angle_physical_mse << '\n';
    if (!output) {
        throw std::runtime_error("Failed to flush test metrics: " + path.string());
    }
}

void print_test_metrics(const TestMetrics& metrics) {
    std::cout << "Test physical MSE | abs(AoA) <= 10 deg: "
              << metrics.low_angle_physical_mse << " (samples="
              << metrics.low_angle_samples << ")\n"
              << "Test physical MSE | abs(AoA) > 10 deg: "
              << metrics.high_angle_physical_mse << " (samples="
              << metrics.high_angle_samples << ")" << std::endl;
}

size_t selected_epoch(const TrainingResult& result) {
    return result.best_epoch != 0 ? result.best_epoch : result.epochs_completed;
}

void write_final_metrics(const std::filesystem::path& output_directory,
                         const FinalMetrics& metrics) {
    std::filesystem::create_directories(output_directory);
    const auto path = output_directory / "final_metrics.csv";
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Failed to write final metrics: " + path.string());
    }

    output << "metric,value\n"
           << "selected_epoch," << metrics.selected_epoch << '\n'
           << std::setprecision(17)
           << "training_physical_mse," << metrics.training_physical_mse << '\n'
           << "validation_physical_mse," << metrics.validation_physical_mse << '\n'
           << "test_physical_mse," << metrics.test_physical_mse << '\n';
    if (!output) {
        throw std::runtime_error("Failed to flush final metrics: " + path.string());
    }
}

void print_final_metrics(const FinalMetrics& metrics) {
    std::cout << "Best epoch: " << metrics.selected_epoch << '\n'
              << "Training physical MSE: " << metrics.training_physical_mse << '\n'
              << "Validation physical MSE: " << metrics.validation_physical_mse << '\n'
              << "Test physical MSE: " << metrics.test_physical_mse << std::endl;
}

size_t parse_size(const std::string& option, const std::string& value) {
    size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (consumed != value.size() || parsed == 0) {
        throw std::invalid_argument(option + " must be a positive integer.");
    }
    return static_cast<size_t>(parsed);
}

uint64_t parse_seed(const std::string& value) {
    size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument("--seed must be a non-negative integer.");
    }
    return static_cast<uint64_t>(parsed);
}

float parse_float(const std::string& option, const std::string& value) {
    size_t consumed = 0;
    const float parsed = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::invalid_argument(option + " must be a finite number.");
    }
    return parsed;
}

CommandLineOptions parse_arguments(int argc, char** argv) {
    CommandLineOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next_value = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + argument + ".");
            }
            return argv[++index];
        };

        if (argument == "--cross-validate") {
            options.cross_validate = true;
        } else if (argument == "--diagnostics") {
            options.diagnostics = true;
        } else if (argument == "--no-diagnostics") {
            options.diagnostics = false;
        } else if (argument == "--verbose-final") {
            options.verbose_final = true;
        } else if (argument == "--quiet-final") {
            options.verbose_final = false;
        } else if (argument == "--help" || argument == "-h") {
            options.show_help = true;
        } else if (argument == "--activation") {
            options.activation = next_value();
        } else if (argument == "--alpha") {
            options.leaky_alpha = parse_float(argument, next_value());
        } else if (argument == "--dropout") {
            options.dropout = parse_float(argument, next_value());
        } else if (argument == "--learning-rate") {
            options.learning_rate = parse_float(argument, next_value());
        } else if (argument == "--physics-weight") {
            options.physics_weight = parse_float(argument, next_value());
        } else if (argument == "--l1-weight") {
            options.l1_weight = parse_float(argument, next_value());
        } else if (argument == "--l2-weight") {
            options.l2_weight = parse_float(argument, next_value());
        } else if (argument == "--gradient-clip") {
            options.gradient_clip = parse_float(argument, next_value());
        } else if (argument == "--epochs") {
            options.epochs = parse_size(argument, next_value());
        } else if (argument == "--batch-size") {
            options.global_batch_size = parse_size(argument, next_value());
        } else if (argument == "--folds") {
            options.folds = parse_size(argument, next_value());
        } else if (argument == "--validation-interval") {
            options.validation_interval = parse_size(argument, next_value());
        } else if (argument == "--histogram-bins") {
            options.histogram_bins = parse_size(argument, next_value());
        } else if (argument == "--seed") {
            options.seed = parse_seed(next_value());
        } else if (argument == "--train-path") {
            options.train_path = next_value();
        } else if (argument == "--test-path") {
            options.test_path = next_value();
        } else if (argument == "--results-dir") {
            options.results_dir = next_value();
        } else if (argument == "--experiment") {
            options.experiment_name = next_value();
        } else if (argument == "--run-name") {
            options.run_name = next_value();
        } else {
            throw std::invalid_argument("Unknown option: " + argument);
        }
    }

    if (options.leaky_alpha < 0.0f || options.learning_rate <= 0.0f ||
        options.physics_weight < 0.0f || options.gradient_clip < 0.0f ||
        options.l1_weight < 0.0f || options.l2_weight < 0.0f ||
        options.dropout < 0.0f || options.dropout >= 1.0f) {
        throw std::invalid_argument(
            "Alpha, learning rate, physics weight, L1/L2 weights, dropout rate, and gradient clip are outside their valid ranges.");
    }
    return options;
}

void print_help() {
    std::cout
        << "Usage: cnn_executable [options]\n"
        << "  --cross-validate       Evaluate the configured model with random K-fold CV\n"
        << "  --diagnostics          Write structured per-epoch training diagnostics (default)\n"
        << "  --no-diagnostics       Disable diagnostics\n"
        << "  --folds N              Number of CV folds (default: 5)\n"
        << "  --validation-interval N Validation frequency (CV default: 10, final default: 1)\n"
        << "  --histogram-bins N     Fixed activation histogram bins (default: 64)\n"
        << "  --epochs N             Epochs per fold/training run (default: 200)\n"
        << "  --batch-size N         Global MPI batch size (default: 257)\n"
        << "  --activation NAME      leakyrelu, relu, tanh, or sigmoid\n"
        << "  --alpha VALUE          LeakyReLU negative slope\n"
        << "  --dropout VALUE        Dropout rate in [0, 1) for the dense head (default: 0)\n"
        << "  --learning-rate VALUE  Adam learning rate (default: 1e-3)\n"
        << "  --physics-weight VALUE SIMM physics weight (default: 0.10)\n"
        << "  --l1-weight VALUE      L1 (Lasso) penalty weight on weight tensors\n"
        << "  --l2-weight VALUE      L2 (Ridge) penalty weight on weight tensors\n"
        << "  --gradient-clip VALUE  Element-wise gradient clipping threshold\n"
        << "  --seed N               Split, shuffle, and initialization seed\n"
        << "  --train-path PATH      Training NPZ path\n"
        << "  --test-path PATH       Untouched test NPZ path\n"
        << "  --results-dir PATH     Diagnostics results root (default: results)\n"
        << "  --experiment NAME      Diagnostics experiment name (default: cnn)\n"
        << "  --run-name NAME        Diagnostics run name (default: run)\n"
        << "  --verbose-final        Print every final-training epoch (default)\n"
        << "  --quiet-final          Suppress final-training epoch lines\n";
}

ModelBlueprint make_blueprint(int kernel_size,
                              int stride,
                              int channels,
                              const std::string& activation,
                              float leaky_alpha,
                              const std::vector<int>& hidden_layers,
                              float dropout = 0.0f) {
    ModelBlueprint blueprint;
    blueprint.feature_layers.push_back(
        Recipes::conv2d(channels, kernel_size, stride, 0));
    blueprint.feature_layers.push_back(
        Recipes::activation(activation, leaky_alpha));
    blueprint.feature_layers.push_back(Recipes::flatten());

    for (int width : hidden_layers) {
        blueprint.head_layers.push_back(Recipes::dense(width));
        blueprint.head_layers.push_back(
            Recipes::activation(activation, leaky_alpha));
        if (dropout > 0.0f) {
            blueprint.head_layers.push_back(Recipes::dropout(dropout));
        }
    }
    blueprint.head_layers.push_back(Recipes::dense(1));
    return blueprint;
}

TrainingConfig make_training_config(const CommandLineOptions& options,
                                    bool early_stopping) {
    TrainingConfig config;
    config.epochs = options.epochs;
    config.global_batch_size = options.global_batch_size;
    config.gradient_clip = options.gradient_clip;
    config.seed = options.seed;
    config.shuffle = true;
    config.validation_interval = options.validation_interval != 0
        ? options.validation_interval
        : (options.cross_validate ? Trainer::kHistoryIntervalEpochs : 1);
    config.diagnostics.enabled = options.diagnostics;
    config.diagnostics.results_root = options.results_dir;
    config.diagnostics.experiment_name = options.experiment_name;
    config.diagnostics.run_name = options.run_name;
    config.diagnostics.histogram_bins = options.histogram_bins;
    config.diagnostics.training_dataset_path = options.train_path;
    config.diagnostics.validation_dataset_path = options.train_path;
    config.early_stopping = early_stopping;
    config.max_overfit_ratio = kOverfitRatio;
    config.restore_best_weights = true;
    return config;
}

TrialConfig make_single_trial(const CommandLineOptions& options,
                              bool early_stopping) {
    TrialConfig trial{
        "single-training",
        make_blueprint(5, 5, 8, options.activation, options.leaky_alpha,
                       {128, 64}, options.dropout),
        Recipes::adam(options.learning_rate, 0.9f, 0.999f, 1e-8f, 0.0f),
        LossConfig{options.physics_weight, options.l1_weight, options.l2_weight},
        make_training_config(options, early_stopping),
        {}};
    if (early_stopping) {
        trial.selected_parameters["validation_fraction"] = "0.10";
    }
    trial.selected_parameters["early_stopping"] = early_stopping ? "true" : "false";
    trial.selected_parameters["early_stopping_min_epochs"] =
        std::to_string(trial.training.early_stopping_min_epochs);
    trial.selected_parameters["early_stopping_patience"] =
        std::to_string(trial.training.early_stopping_patience);
    trial.selected_parameters["max_overfit_ratio"] = "0.15";
    trial.selected_parameters["restore_best_weights"] = "true";
    return trial;
}

TrainedRun train_and_validate(const TrialConfig& config,
                              const Dataset& training_dataset,
                              std::span<const size_t> training_indices,
                              const Dataset& validation_dataset,
                              std::span<const size_t> validation_indices,
                              const NormalizationStats& normalization,
                              MPI_Comm communicator,
                              bool verbose,
                              const TrainingRunContext& run_context) {
    ModelFactory factory;
    auto model = factory.build(config, training_dataset.sdf_height(),
                               training_dataset.sdf_width(),
                               training_dataset.scalar_features(),
                               config.training.seed, communicator);
    Trainer trainer(communicator);
    TrainingResult result = trainer.fit(
        *model, training_dataset, training_indices, validation_dataset,
        validation_indices, normalization, config.loss, config.training,
        config.training.seed, verbose, run_context, &config);
    return TrainedRun{std::move(model), std::move(result)};
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try {
        const CommandLineOptions options = parse_arguments(argc, argv);
        if (options.show_help) {
            if (rank == 0) {
                print_help();
            }
            MPI_Finalize();
            return 0;
        }

        Dataset training_dataset(options.train_path);
        if (!options.cross_validate) {
            const DatasetSplit split =
                make_training_validation_split(training_dataset, options.seed);
            const TrialConfig config = make_single_trial(options, true);
            const NormalizationStats normalization =
                training_dataset.fit_normalization(split.training);
            TrainingRunContext run_context;
            run_context.random_seed = config.training.seed;
            run_context.training_dataset_path = options.train_path;
            run_context.validation_dataset_path = options.train_path;
            TrainedRun trained = train_and_validate(
                config, training_dataset, split.training, training_dataset,
                split.validation, normalization, MPI_COMM_WORLD,
                options.verbose_final, run_context);
            Dataset test_dataset(options.test_path);
            const auto test_indices = test_dataset.all_indices();
            const TestMetrics test_metrics = evaluate_test_metrics(
                *trained.model, test_dataset, test_indices,
                normalization, config.training.global_batch_size,
                MPI_COMM_WORLD);
            if (rank == 0) {
                write_test_metrics(
                    diagnostics_run_directory(config.training.diagnostics,
                                              run_context),
                    test_metrics);
                const FinalMetrics final_metrics{
                    selected_epoch(trained.result),
                    trained.result.training_mse,
                    trained.result.validation_mse,
                    test_metrics.overall_physical_mse};
                write_final_metrics(
                    diagnostics_run_directory(config.training.diagnostics,
                                              run_context),
                    final_metrics);
                print_final_metrics(final_metrics);
                print_test_metrics(test_metrics);
            }
        } else {
            auto splitter = std::make_shared<RandomKFold>(
                options.folds, true, options.seed);
            auto runner = std::make_shared<CNNTrialRunner>(MPI_COMM_WORLD);
            CrossValidator validator(training_dataset, splitter, runner,
                                     MPI_COMM_WORLD, true);
            const TrialConfig config = make_single_trial(options, false);
            const CandidateResult result = validator.evaluate(config);

            if (rank == 0) {
                std::cout << "Cross-validation physical MSE: "
                          << result.mean_validation_mse << " +/- "
                          << result.validation_stddev << std::endl;
                for (const FoldMetrics& fold : result.folds) {
                    for (const EpochMetrics& point : fold.history) {
                        std::cout << "Fold " << (fold.fold + 1)
                                  << " | Epoch " << point.epoch
                                  << " | Train objective: "
                                  << point.training_objective
                                  << " | Validation physical MSE: "
                                  << point.validation_mse << std::endl;
                    }
                }
            }

            TrialConfig final_config = config;
            if (options.validation_interval == 0) {
                final_config.training.validation_interval = 1;
            }
            const auto training_indices = training_dataset.all_indices();
            const NormalizationStats normalization =
                training_dataset.fit_normalization(training_indices);
            TrainingRunContext final_context;
            final_context.final_subdirectory = options.diagnostics;
            final_context.random_seed = final_config.training.seed;
            final_context.training_dataset_path = options.train_path;
            final_context.validation_dataset_path = options.train_path;
            TrainedRun final_trained = train_and_validate(
                final_config, training_dataset, training_indices,
                training_dataset, training_indices, normalization, MPI_COMM_WORLD,
                options.verbose_final, final_context);
            Dataset test_dataset(options.test_path);
            const auto test_indices = test_dataset.all_indices();
            const TestMetrics test_metrics = evaluate_test_metrics(
                *final_trained.model, test_dataset, test_indices, normalization,
                final_config.training.global_batch_size, MPI_COMM_WORLD);
            if (rank == 0) {
                write_test_metrics(
                    diagnostics_run_directory(final_config.training.diagnostics,
                                              final_context),
                    test_metrics);
                const FinalMetrics final_metrics{
                    selected_epoch(final_trained.result),
                    final_trained.result.training_mse,
                    final_trained.result.validation_mse,
                    test_metrics.overall_physical_mse};
                write_final_metrics(
                    diagnostics_run_directory(final_config.training.diagnostics,
                                              final_context),
                    final_metrics);
                print_final_metrics(final_metrics);
                print_test_metrics(test_metrics);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "CNN execution failed on rank " << rank << ": "
                  << error.what() << std::endl;
        int world_size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (world_size > 1) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
