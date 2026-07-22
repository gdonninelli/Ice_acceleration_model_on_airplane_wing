#include "src/data/Dataset.hpp"
#include "src/model/ModelFactory.hpp"
#include "src/training/Trainer.hpp"
#include "src/tuning/CrossValidator.hpp"
#include "src/tuning/SearchSpace.hpp"
#include "src/tuning/TrialConfig.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
struct CommandLineOptions {
    bool cross_validate = false;
    bool show_help = false;
    std::string activation = "leakyrelu";
    float leaky_alpha = 0.05f;
    float learning_rate = 1e-5f;
    float physics_weight = 0.25f;
    float gradient_clip = 1.0f;
    size_t epochs = 100;
    size_t global_batch_size = 64;
    size_t folds = 5;
    uint64_t seed = 42;
    std::string train_path = "dataset/cnn_dataset_train.npz";
    std::string test_path = "dataset/cnn_dataset_test.npz";
};

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
        } else if (argument == "--help" || argument == "-h") {
            options.show_help = true;
        } else if (argument == "--activation") {
            options.activation = next_value();
        } else if (argument == "--alpha") {
            options.leaky_alpha = parse_float(argument, next_value());
        } else if (argument == "--learning-rate") {
            options.learning_rate = parse_float(argument, next_value());
        } else if (argument == "--physics-weight") {
            options.physics_weight = parse_float(argument, next_value());
        } else if (argument == "--gradient-clip") {
            options.gradient_clip = parse_float(argument, next_value());
        } else if (argument == "--epochs") {
            options.epochs = parse_size(argument, next_value());
        } else if (argument == "--batch-size") {
            options.global_batch_size = parse_size(argument, next_value());
        } else if (argument == "--folds") {
            options.folds = parse_size(argument, next_value());
        } else if (argument == "--seed") {
            options.seed = parse_seed(next_value());
        } else if (argument == "--train-path") {
            options.train_path = next_value();
        } else if (argument == "--test-path") {
            options.test_path = next_value();
        } else {
            throw std::invalid_argument("Unknown option: " + argument);
        }
    }

    if (options.leaky_alpha < 0.0f || options.learning_rate <= 0.0f ||
        options.physics_weight < 0.0f || options.gradient_clip < 0.0f) {
        throw std::invalid_argument(
            "Alpha, learning rate, physics weight, and gradient clip are outside their valid ranges.");
    }
    return options;
}

void print_help() {
    std::cout
        << "Usage: cnn_executable [options]\n"
        << "  --cross-validate       Evaluate the configured model with random K-fold CV\n"
        << "  --folds N              Number of CV folds (default: 5)\n"
        << "  --epochs N             Epochs per fold/training run (default: 100)\n"
        << "  --batch-size N         Global MPI batch size (default: 64)\n"
        << "  --activation NAME      leakyrelu, relu, tanh, or sigmoid\n"
        << "  --alpha VALUE          LeakyReLU negative slope\n"
        << "  --learning-rate VALUE  Adam learning rate for ordinary training\n"
        << "  --physics-weight VALUE SIMM physics weight\n"
        << "  --gradient-clip VALUE  Element-wise gradient clipping threshold\n"
        << "  --seed N               Split, shuffle, and initialization seed\n"
        << "  --train-path PATH      Training NPZ path\n"
        << "  --test-path PATH       Untouched test NPZ path\n";
}

ModelBlueprint make_blueprint(int kernel_size,
                              int stride,
                              int channels,
                              const std::string& activation,
                              float leaky_alpha,
                              const std::vector<int>& hidden_layers) {
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
    }
    blueprint.head_layers.push_back(Recipes::dense(1));
    return blueprint;
}

TrainingConfig make_training_config(const CommandLineOptions& options) {
    return TrainingConfig{options.epochs,
                          options.global_batch_size,
                          options.gradient_clip,
                          options.seed,
                          true};
}

TrialConfig make_single_trial(const CommandLineOptions& options) {
    return TrialConfig{
        "single-training",
        make_blueprint(5, 5, 8, options.activation, options.leaky_alpha,
                       {128, 64}),
        Recipes::adam(options.learning_rate),
        LossConfig{options.physics_weight},
        make_training_config(options),
        {}};
}

TrainingResult train_and_test(const TrialConfig& config,
                              const Dataset& training_dataset,
                              const Dataset& test_dataset,
                              MPI_Comm communicator,
                              bool verbose) {
    const auto training_indices = training_dataset.all_indices();
    const auto test_indices = test_dataset.all_indices();
    const NormalizationStats normalization =
        training_dataset.fit_normalization(training_indices);
    ModelFactory factory;
    auto model = factory.build(config, training_dataset.sdf_height(),
                               training_dataset.sdf_width(),
                               training_dataset.scalar_features(),
                               config.training.seed, communicator);
    Trainer trainer(communicator);
    return trainer.fit(*model, training_dataset, training_indices, test_dataset,
                       test_indices, normalization, config.loss, config.training,
                       config.training.seed, verbose);
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
            Dataset test_dataset(options.test_path);
            const TrialConfig config = make_single_trial(options);
            const TrainingResult result = train_and_test(
                config, training_dataset, test_dataset, MPI_COMM_WORLD, true);
            if (rank == 0) {
                std::cout << "Final test physical MSE: "
                          << result.validation_mse << std::endl;
            }
        } else {
            auto splitter = std::make_shared<RandomKFold>(
                options.folds, true, options.seed);
            auto runner = std::make_shared<CNNTrialRunner>(MPI_COMM_WORLD);
            CrossValidator validator(training_dataset, splitter, runner,
                                     MPI_COMM_WORLD, true);
            const TrialConfig config = make_single_trial(options);
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

            Dataset test_dataset(options.test_path);
            const TrainingResult final_result = train_and_test(
                config, training_dataset, test_dataset, MPI_COMM_WORLD,
                true);
            if (rank == 0) {
                std::cout << "Final test physical MSE: "
                          << final_result.validation_mse << std::endl;
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
