#include "core/Loss.hpp"
#include "data/Dataset.hpp"
#include "model/ModelFactory.hpp"
#include "training/Trainer.hpp"
#include "training/TrainingDiagnostics.hpp"
#include "tuning/TrialConfig.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mpi.h>
#include <numbers>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr double kValidationFraction = 0.10;
constexpr double kOverfitRatio = 0.15;
constexpr double kAoAThresholdRadians =
    10.0 * std::numbers::pi / 180.0;

struct StageSpec {
    const char* name;
    size_t epochs;
    size_t global_batch_size;
    uint64_t seed;
    BatchConstruction batch_construction;
    EarlyStoppingPolicy early_stopping_policy;
};

constexpr StageSpec kStages[] = {
    {"trial-1", 200, 64, 42, BatchConstruction::RangeTail,
     EarlyStoppingPolicy::FirstRatioExceeded},
    {"trial-2", 200, 64, 42, BatchConstruction::Balanced,
     EarlyStoppingPolicy::FirstRatioExceeded},
    {"trial-3", 200, 257, 42, BatchConstruction::Balanced,
     EarlyStoppingPolicy::FirstRatioExceeded},
    {"trial-4", 834, 257, 42, BatchConstruction::Balanced,
     EarlyStoppingPolicy::FirstRatioExceeded},
    {"trial-4-seed-0", 834, 257, 0, BatchConstruction::Balanced,
     EarlyStoppingPolicy::FirstRatioExceeded},
    {"trial-4-seed-1", 834, 257, 1, BatchConstruction::Balanced,
     EarlyStoppingPolicy::FirstRatioExceeded},
    {"trial-4-seed-0b", 834, 257, 0, BatchConstruction::Balanced,
     EarlyStoppingPolicy::Patience},
    {"trial-4-seed-1b", 834, 257, 1, BatchConstruction::Balanced,
     EarlyStoppingPolicy::Patience},
    {"trial-4-seed-42b", 834, 257, 42, BatchConstruction::Balanced,
     EarlyStoppingPolicy::Patience},
    {"trial-5", 1200, 257, 42, BatchConstruction::Balanced,
     EarlyStoppingPolicy::Patience},
    {"production", 2500, 257, 42, BatchConstruction::Balanced,
     EarlyStoppingPolicy::Patience},
};

struct Options {
    std::string stage;
    std::string run_name;
    std::string train_path = "dataset/cnn_dataset_train.npz";
    std::string test_path = "dataset/cnn_dataset_test.npz";
    std::string results_dir = "results";
    bool diagnostics = true;
    bool verbose = true;
    bool smoke = false;
    bool help = false;
};

struct DatasetSplit {
    std::vector<size_t> training;
    std::vector<size_t> validation;
};

struct Partition {
    size_t offset = 0;
    size_t count = 0;
};

struct TestMetrics {
    double overall = std::numeric_limits<double>::quiet_NaN();
    double low_angle = std::numeric_limits<double>::quiet_NaN();
    double high_angle = std::numeric_limits<double>::quiet_NaN();
    size_t overall_samples = 0;
    size_t low_angle_samples = 0;
    size_t high_angle_samples = 0;
};

const StageSpec& find_stage(const std::string& name) {
    const auto found = std::find_if(
        std::begin(kStages), std::end(kStages),
        [&](const StageSpec& stage) { return name == stage.name; });
    if (found == std::end(kStages)) {
        throw std::invalid_argument("Unknown stage: " + name);
    }
    return *found;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + argument);
            }
            return argv[++index];
        };
        if (argument == "--stage") options.stage = next();
        else if (argument == "--run-name") options.run_name = next();
        else if (argument == "--train-path") options.train_path = next();
        else if (argument == "--test-path") options.test_path = next();
        else if (argument == "--results-dir") options.results_dir = next();
        else if (argument == "--diagnostics") options.diagnostics = true;
        else if (argument == "--no-diagnostics") options.diagnostics = false;
        else if (argument == "--quiet") options.verbose = false;
        else if (argument == "--smoke") options.smoke = true;
        else if (argument == "--help" || argument == "-h") options.help = true;
        else throw std::invalid_argument("Unknown option: " + argument);
    }
    if (!options.help && options.stage.empty()) {
        throw std::invalid_argument("--stage is required");
    }
    if (!options.help && options.run_name.empty()) {
        throw std::invalid_argument("--run-name is required");
    }
    return options;
}

void print_help() {
    std::cout << "Usage: re_tuning --stage NAME --run-name NAME [options]\n\n"
              << "Stages:\n";
    for (const StageSpec& stage : kStages) {
        std::cout << "  " << stage.name << '\n';
    }
    std::cout << "\nOptions:\n"
              << "  --train-path PATH    Training NPZ\n"
              << "  --test-path PATH     Untouched test NPZ\n"
              << "  --results-dir PATH   Results root (default: results)\n"
              << "  --no-diagnostics     Disable structured diagnostics\n"
              << "  --quiet              Suppress per-epoch output\n"
              << "  --smoke              Run two epochs without early stopping\n";
}

ModelBlueprint make_large_blueprint() {
    ModelBlueprint blueprint;
    blueprint.feature_layers = {
        Recipes::conv2d(8, 5, 5, 0),
        Recipes::activation("leakyrelu", 0.05f),
        Recipes::flatten()};
    for (int width : {1024, 512, 256, 128}) {
        blueprint.head_layers.push_back(Recipes::dense(width));
        blueprint.head_layers.push_back(
            Recipes::activation("leakyrelu", 0.05f));
    }
    blueprint.head_layers.push_back(Recipes::dense(1));
    return blueprint;
}

TrialConfig make_trial(const StageSpec& stage,
                       const Options& options) {
    TrainingConfig training;
    training.epochs = options.smoke ? 2 : stage.epochs;
    training.global_batch_size = stage.global_batch_size;
    training.gradient_clip = 1.0f;
    training.seed = stage.seed;
    training.shuffle = true;
    training.validation_interval = 1;
    training.early_stopping = !options.smoke;
    const bool first_crossing = stage.early_stopping_policy ==
        EarlyStoppingPolicy::FirstRatioExceeded;
    training.early_stopping_min_epochs = first_crossing ? 0 : 20;
    training.early_stopping_patience = first_crossing ? 1 : 20;
    training.max_overfit_ratio = kOverfitRatio;
    training.restore_best_weights = true;
    training.batch_construction = stage.batch_construction;
    training.early_stopping_policy = stage.early_stopping_policy;
    training.diagnostics.enabled = options.diagnostics;
    training.diagnostics.results_root = options.results_dir;
    training.diagnostics.experiment_name = "re-tuning";
    training.diagnostics.run_name = options.run_name;
    training.diagnostics.histogram_bins = 64;
    training.diagnostics.training_dataset_path = options.train_path;
    training.diagnostics.validation_dataset_path = options.train_path;

    TrialConfig trial{
        "large-topology-" + std::string(stage.name),
        make_large_blueprint(),
        Recipes::adam(1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f),
        LossConfig{0.10f, 0.0f, 0.0f},
        training,
        {}};
    trial.selected_parameters["stage"] = stage.name;
    trial.selected_parameters["topology"] =
        "conv5x5-dense-1024-512-256-128";
    trial.selected_parameters["batch_construction"] =
        batch_construction_name(training.batch_construction);
    trial.selected_parameters["early_stopping_policy"] =
        early_stopping_policy_name(training.early_stopping_policy);
    trial.selected_parameters["early_stopping_min_epochs"] =
        std::to_string(training.early_stopping_min_epochs);
    trial.selected_parameters["early_stopping_patience"] =
        std::to_string(training.early_stopping_patience);
    trial.selected_parameters["max_overfit_ratio"] = "0.15";
    trial.selected_parameters["restore_best_weights"] = "true";
    trial.selected_parameters["validation_fraction"] = "0.10";
    return trial;
}

DatasetSplit make_split(const Dataset& dataset, uint64_t seed) {
    std::vector<size_t> shuffled = dataset.all_indices();
    if (shuffled.size() < 2) {
        throw std::invalid_argument("Training dataset requires at least two samples");
    }
    std::mt19937_64 generator(seed);
    std::shuffle(shuffled.begin(), shuffled.end(), generator);
    const size_t validation_count = std::max(
        size_t{1}, static_cast<size_t>(shuffled.size() * kValidationFraction));
    return {
        std::vector<size_t>(shuffled.begin() + validation_count, shuffled.end()),
        std::vector<size_t>(shuffled.begin(), shuffled.begin() + validation_count)};
}

Partition partition(size_t count, int rank, int world_size) {
    const size_t processes = static_cast<size_t>(world_size);
    const size_t base = count / processes;
    const size_t remainder = count % processes;
    const size_t rank_index = static_cast<size_t>(rank);
    return {
        rank_index * base + std::min(rank_index, remainder),
        base + (rank_index < remainder ? 1 : 0)};
}

TestMetrics evaluate_test(CNNModel& model,
                          const Dataset& dataset,
                          const NormalizationStats& normalization,
                          size_t batch_size) {
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const auto indices = dataset.all_indices();
    double local_sums[3]{0.0, 0.0, 0.0};
    unsigned long long local_counts[3]{0, 0, 0};
    std::string local_error;
    try {
        for (size_t offset = 0; offset < indices.size(); offset += batch_size) {
            const size_t global_count =
                std::min(batch_size, indices.size() - offset);
            const Partition local = partition(global_count, rank, world_size);
            if (local.count == 0) continue;
            const std::span<const size_t> local_indices(
                indices.data() + offset + local.offset, local.count);
            const DataBatch batch = dataset.make_batch(local_indices, normalization);
            const auto predictions = model.predict(batch.sdf, batch.scalars);
            const float* prediction = predictions->get_data();
            const float* target = batch.targets->get_data();
            const float* angle = batch.alpha_radians->get_data();
            for (size_t sample = 0; sample < local.count; ++sample) {
                const double error =
                    (static_cast<double>(prediction[sample]) - target[sample]) *
                    static_cast<double>(
                        static_cast<float>(normalization.target_std));
                const double squared = error * error;
                local_sums[0] += squared;
                ++local_counts[0];
                const size_t bucket =
                    std::abs(static_cast<double>(angle[sample])) >
                            kAoAThresholdRadians
                        ? 2
                        : 1;
                local_sums[bucket] += squared;
                ++local_counts[bucket];
            }
        }
    } catch (const std::exception& error) {
        local_error = error.what();
    } catch (...) {
        local_error = "unknown test-evaluation error";
    }
    int local_failed = local_error.empty() ? 0 : 1;
    int any_failed = 0;
    MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (any_failed != 0) {
        throw std::runtime_error(local_error.empty()
            ? "Test evaluation failed on another MPI rank"
            : "Test evaluation failed: " + local_error);
    }
    double global_sums[3]{};
    unsigned long long global_counts[3]{};
    MPI_Allreduce(local_sums, global_sums, 3, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(local_counts, global_counts, 3, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    if (global_counts[0] == 0 || global_counts[1] == 0 ||
        global_counts[2] == 0) {
        throw std::runtime_error("Test evaluation produced an empty metric subset");
    }
    const auto mean = [&](size_t index) {
        return global_sums[index] / static_cast<double>(global_counts[index]);
    };
    return {
        mean(0), mean(1), mean(2),
        static_cast<size_t>(global_counts[0]),
        static_cast<size_t>(global_counts[1]),
        static_cast<size_t>(global_counts[2])};
}

void write_metrics(const std::filesystem::path& directory,
                   const TrainingResult& result,
                   const TestMetrics& test) {
    std::filesystem::create_directories(directory);
    {
        const auto path = directory / "final_metrics.csv";
        std::ofstream output(path, std::ios::trunc);
        if (!output) throw std::runtime_error("Cannot write " + path.string());
        const size_t selected = result.best_epoch != 0
            ? result.best_epoch
            : result.epochs_completed;
        output << "metric,value\n"
               << "selected_epoch," << selected << '\n'
               << std::setprecision(17)
               << "training_physical_mse," << result.training_mse << '\n'
               << "validation_physical_mse," << result.validation_mse << '\n'
               << "test_physical_mse," << test.overall << '\n';
        output.flush();
        if (!output) throw std::runtime_error("Cannot flush " + path.string());
    }
    {
        const auto path = directory / "test_metrics.csv";
        std::ofstream output(path, std::ios::trunc);
        if (!output) throw std::runtime_error("Cannot write " + path.string());
        output << "subset,samples,physical_mse\n" << std::setprecision(17)
               << "overall," << test.overall_samples << ',' << test.overall << '\n'
               << "abs_aoa_le_10_deg," << test.low_angle_samples << ','
               << test.low_angle << '\n'
               << "abs_aoa_gt_10_deg," << test.high_angle_samples << ','
               << test.high_angle << '\n';
        output.flush();
        if (!output) throw std::runtime_error("Cannot flush " + path.string());
    }
}

void ensure_output_is_new(const TrainingConfig& config,
                          const TrainingRunContext& context) {
    const auto directory = diagnostics_run_directory(config.diagnostics, context);
    int collision = 0;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0 && std::filesystem::exists(directory) &&
        !std::filesystem::is_empty(directory)) {
        collision = 1;
    }
    MPI_Bcast(&collision, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (collision != 0) {
        throw std::runtime_error("Output directory already exists and is not empty: " +
                                 directory.string());
    }
}

void print_configuration(const StageSpec& stage,
                         const TrialConfig& trial,
                         const Options& options) {
    std::cout << "Re-training stage: " << stage.name << '\n'
              << "Run name: " << options.run_name << '\n'
              << "Topology: conv5x5, channels 8, dense 1024-512-256-128-1\n"
              << "Adam learning rate: 1e-3\n"
              << "Physics weight: 0.10\n"
              << "Dropout/L1/L2: 0/0/0\n"
              << "Epoch cap: " << trial.training.epochs << '\n'
              << "Global batch size: " << trial.training.global_batch_size << '\n'
              << "Batch construction: "
              << batch_construction_name(trial.training.batch_construction) << '\n'
              << "Early stopping: "
              << (trial.training.early_stopping ? "enabled" : "disabled") << '\n'
              << "Early stopping policy: "
              << early_stopping_policy_name(
                     trial.training.early_stopping_policy) << '\n'
              << "Seed: " << trial.training.seed << std::endl;
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        const Options options = parse_options(argc, argv);
        if (options.help) {
            if (rank == 0) print_help();
            MPI_Finalize();
            return 0;
        }
        const StageSpec& stage = find_stage(options.stage);
        const TrialConfig trial = make_trial(stage, options);
        if (rank == 0) print_configuration(stage, trial, options);

        TrainingRunContext context;
        context.random_seed = trial.training.seed;
        context.training_dataset_path = options.train_path;
        context.validation_dataset_path = options.train_path;
        ensure_output_is_new(trial.training, context);

        Dataset training_dataset(options.train_path);
        const DatasetSplit split = make_split(training_dataset, trial.training.seed);
        const NormalizationStats normalization =
            training_dataset.fit_normalization(split.training);
        ModelFactory factory;
        auto model = factory.build(
            trial, training_dataset.sdf_height(), training_dataset.sdf_width(),
            training_dataset.scalar_features(), trial.training.seed,
            MPI_COMM_WORLD);
        Trainer trainer(MPI_COMM_WORLD);
        const TrainingResult result = trainer.fit(
            *model, training_dataset, split.training, training_dataset,
            split.validation, normalization, trial.loss, trial.training,
            trial.training.seed, options.verbose, context, &trial);

        Dataset test_dataset(options.test_path);
        const TestMetrics test = evaluate_test(
            *model, test_dataset, normalization,
            trial.training.global_batch_size);
        if (rank == 0) {
            const auto directory = diagnostics_run_directory(
                trial.training.diagnostics, context);
            model->export_weights((directory / "model_weights.bin").string());
            write_metrics(directory, result, test);
            const size_t selected = result.best_epoch != 0
                ? result.best_epoch
                : result.epochs_completed;
            std::cout << "Completed epoch: " << result.epochs_completed << '\n'
                      << "Selected epoch: " << selected << '\n'
                      << "Training physical MSE: " << result.training_mse << '\n'
                      << "Validation physical MSE: " << result.validation_mse << '\n'
                      << "Test physical MSE: " << test.overall << '\n'
                      << "Results: " << directory << std::endl;
        }
    } catch (const std::exception& error) {
        std::cerr << "re_tuning failed on rank " << rank << ": "
                  << error.what() << std::endl;
        int world_size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (world_size > 1) MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        return 1;
    }
    MPI_Finalize();
    return 0;
}
