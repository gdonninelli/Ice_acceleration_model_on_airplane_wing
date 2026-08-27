// Activation-function tuning experiment.
//
// The architecture and training budget are fixed while the activation used by
// the convolutional trunk and dense head is varied. Candidate selection uses
// the physical-unit validation MSE returned by CrossValidator.

#include "data/Dataset.hpp"
#include "model/ModelFactory.hpp"
#include "training/Trainer.hpp"
#include "tuning/CrossValidator.hpp"
#include "tuning/SearchSpace.hpp"
#include "tuning/TrialConfig.hpp"
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr size_t kDefaultFolds = 5;
constexpr size_t kDefaultEpochs = 100;
constexpr size_t kDefaultBatchSize = 64;
constexpr size_t kDefaultValidationInterval = 10;
constexpr uint64_t kDefaultSeed = 42;
constexpr float kLearningRate = 1e-5f;
constexpr float kPhysicsWeight = 0.1f;
constexpr size_t kHistogramBins = 64;

struct ActivationChoice {
    std::string name;
    float alpha = 0.05f;
};

ModelBlueprint make_blueprint(const ActivationChoice& activation) {
    ModelBlueprint blueprint;
    blueprint.feature_layers = {
        Recipes::conv2d(8, 5, 5, 0),
        Recipes::activation(activation.name, activation.alpha),
        Recipes::flatten()};

    for (int width : {1024, 512, 256, 128}) {
        blueprint.head_layers.push_back(Recipes::dense(width));
        blueprint.head_layers.push_back(
            Recipes::activation(activation.name, activation.alpha));
    }
    blueprint.head_layers.push_back(Recipes::dense(1));
    return blueprint;
}

size_t parse_size(const std::string& option, const std::string& value) {
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument(option + " must be a positive integer.");
    }
    size_t consumed = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(option + " must be a positive integer.");
    }
    if (consumed != value.size() || parsed == 0 ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        throw std::invalid_argument(option + " must be a positive integer.");
    }
    return static_cast<size_t>(parsed);
}

uint64_t parse_seed(const std::string& value) {
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument("--seed must be a non-negative integer.");
    }
    size_t consumed = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument("--seed must be a non-negative integer.");
    }
    if (consumed != value.size()) {
        throw std::invalid_argument("--seed must be a non-negative integer.");
    }
    return static_cast<uint64_t>(parsed);
}

struct ProgramOptions {
    size_t folds = kDefaultFolds;
    size_t epochs = kDefaultEpochs;
    size_t batch_size = kDefaultBatchSize;
    size_t validation_interval = kDefaultValidationInterval;
    uint64_t seed = kDefaultSeed;
    std::string train_path = "dataset/cnn_dataset_train.npz";
    std::string test_path = "dataset/cnn_dataset_test.npz";
    std::string results_dir =
        "results/cross_validation/activation-function-tuning";
    bool diagnostics = false;
    size_t histogram_bins = kHistogramBins;
    bool smoke = false;
    bool help = false;
};

ProgramOptions parse_options(int argc, char** argv) {
    ProgramOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next_value = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + argument + ".");
            }
            return argv[++index];
        };

        if (argument == "--folds") {
            options.folds = parse_size(argument, next_value());
        } else if (argument == "--epochs") {
            options.epochs = parse_size(argument, next_value());
        } else if (argument == "--batch-size") {
            options.batch_size = parse_size(argument, next_value());
        } else if (argument == "--validation-interval") {
            options.validation_interval = parse_size(argument, next_value());
        } else if (argument == "--seed") {
            options.seed = parse_seed(next_value());
        } else if (argument == "--train-path") {
            options.train_path = next_value();
        } else if (argument == "--test-path") {
            options.test_path = next_value();
        } else if (argument == "--results-dir") {
            options.results_dir = next_value();
        } else if (argument == "--diagnostic" || argument == "--diagnostics") {
            options.diagnostics = true;
        } else if (argument == "--no-diagnostic" || argument == "--no-diagnostics") {
            options.diagnostics = false;
        } else if (argument == "--histogram-bins") {
            options.histogram_bins = parse_size(argument, next_value());
        } else if (argument == "--smoke") {
            options.smoke = true;
            options.folds = 2;
            options.epochs = 2;
            options.validation_interval = 1;
        } else if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else {
            throw std::invalid_argument("Unknown option: " + argument);
        }
    }
    return options;
}

void validate_options(const ProgramOptions& options) {
    if (options.folds < 2) {
        throw std::invalid_argument("--folds must be at least 2.");
    }
    if (options.validation_interval > options.epochs) {
        throw std::invalid_argument(
            "--validation-interval cannot exceed --epochs.");
    }
    if (options.train_path.empty() || options.test_path.empty()) {
        throw std::invalid_argument("Dataset paths must not be empty.");
    }
    if (options.results_dir.empty()) {
        throw std::invalid_argument("--results-dir must not be empty.");
    }
}

void print_help() {
    std::cout
        << "Activation-function tuning for the CNN wing ice model\n\n"
        << "Usage: activation-function-tuning [options]\n\n"
        << "Evaluates 6 activation candidates: tanh, sigmoid, relu, and "
           "LeakyReLU alphas 0.01, 0.05, 0.1.\n\n"
        << "Options:\n"
        << "  --folds N                CV folds (default: 5)\n"
        << "  --epochs N               Epochs per fold (default: 100)\n"
        << "  --batch-size N           Global MPI batch size (default: 64)\n"
        << "  --validation-interval N  Validation/history frequency (default: 10)\n"
        << "  --seed N                 Split, shuffle, and initialization seed (default: 42)\n"
        << "  --train-path PATH        Training NPZ path\n"
        << "  --test-path PATH         Untouched test NPZ path\n"
        << "  --results-dir PATH       Diagnostics and CSV output directory\n"
        << "  --diagnostic             Enable structured training diagnostics\n"
        << "  --diagnostics            Alias for --diagnostic\n"
        << "  --no-diagnostic          Disable diagnostics (default)\n"
        << "  --histogram-bins N       Activation histogram bins (default: 64)\n"
        << "  --smoke                  Use 2 folds, 2 epochs, and interval 1\n"
        << "  --help, -h               Show this message\n\n"
        << "Fixed settings: conv5x5-dense-1024-512-256-128, Adam lr=1e-5, "
           "physics weight=0.1, L1/L2=0, dropout=0.\n";
}

TrialConfig make_base_config(const ProgramOptions& options) {
    TrainingConfig training;
    training.epochs = options.epochs;
    training.global_batch_size = options.batch_size;
    training.gradient_clip = 1.0f;
    training.seed = options.seed;
    training.shuffle = true;
    training.validation_interval = options.validation_interval;
    training.diagnostics.enabled = options.diagnostics;
    training.diagnostics.results_root = options.results_dir;
    training.diagnostics.experiment_name = "activation-function-tuning";
    training.diagnostics.run_name = "search";
    training.diagnostics.histogram_bins = options.histogram_bins;
    training.diagnostics.training_dataset_path = options.train_path;
    training.diagnostics.validation_dataset_path = options.train_path;

    return TrialConfig{"activation-function-search",
                       make_blueprint({"relu", 0.05f}),
                       Recipes::adam(kLearningRate),
                       LossConfig{kPhysicsWeight, 0.0f, 0.0f},
                       training,
                       {}};
}

ParameterGrid make_grid(const ProgramOptions& options) {
    ParameterGrid grid(make_base_config(options));
    grid.add_choice<ActivationChoice>(
        "activation",
        {{"tanh", {"tanh", 0.05f}},
         {"sigmoid", {"sigmoid", 0.05f}},
         {"relu", {"relu", 0.05f}},
         {"leakyrelu-alpha-0.01", {"leakyrelu", 0.01f}},
         {"leakyrelu-alpha-0.05", {"leakyrelu", 0.05f}},
         {"leakyrelu-alpha-0.1", {"leakyrelu", 0.1f}}},
        [](TrialConfig& trial, const ActivationChoice& activation) {
            trial.model = make_blueprint(activation);
        });
    return grid;
}

void write_results(const std::string& path, const SearchResult& result) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot open " + path + " for writing.");
    }

    output << std::setprecision(10)
           << "candidate,activation,fold,training_samples,validation_samples,"
              "training_objective,validation_mse\n";
    for (const CandidateResult& candidate : result.candidates) {
        const auto activation = candidate.config.selected_parameters.find("activation");
        const std::string activation_name =
            activation == candidate.config.selected_parameters.end()
                ? ""
                : activation->second;
        if (!candidate.success) {
            output << '"' << candidate.config.name << "\"," << activation_name
                   << ",FAILED,,,,\n";
            continue;
        }
        for (const FoldMetrics& fold : candidate.folds) {
            output << '"' << candidate.config.name << "\"," << activation_name
                   << ',' << fold.fold << ',' << fold.training_samples << ','
                   << fold.validation_samples << ',' << fold.training_objective
                   << ',' << fold.validation_mse << '\n';
        }
    }
}

void write_history(const std::string& path, const SearchResult& result) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot open " + path + " for writing.");
    }

    output << std::setprecision(10)
           << "candidate,activation,fold,epoch,training_objective,validation_mse\n";
    for (const CandidateResult& candidate : result.candidates) {
        if (!candidate.success) {
            continue;
        }
        const auto activation = candidate.config.selected_parameters.find("activation");
        const std::string activation_name =
            activation == candidate.config.selected_parameters.end()
                ? ""
                : activation->second;
        for (const FoldMetrics& fold : candidate.folds) {
            for (const EpochMetrics& point : fold.history) {
                output << '"' << candidate.config.name << "\"," << activation_name
                       << ',' << fold.fold << ',' << point.epoch << ','
                       << point.training_objective << ',' << point.validation_mse
                       << '\n';
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int status = 0;
    try {
        const ProgramOptions options = parse_options(argc, argv);
        if (options.help) {
            if (rank == 0) {
                print_help();
            }
            MPI_Finalize();
            return 0;
        }
        validate_options(options);

        if (rank == 0) {
            std::cout << "========================================================\n"
                      << "       ACTIVATION-FUNCTION TUNING EXPERIMENT\n"
                      << "========================================================\n"
                      << "Candidates:       6\n"
                      << "Architecture:     conv5x5-dense-1024-512-256-128\n"
                      << "Optimizer:        Adam (lr = " << kLearningRate << ")\n"
                      << "Physics weight:    " << kPhysicsWeight << "\n"
                      << "Folds:             " << options.folds << "\n"
                      << "Epochs:            " << options.epochs << "\n"
                      << "Batch size:        " << options.batch_size << "\n"
                      << "Diagnostics:       "
                      << (options.diagnostics ? "on" : "off") << "\n"
                      << "MPI ranks:         " << world_size << "\n"
                      << "========================================================\n";
        }

        Dataset training_dataset(options.train_path);
        ParameterGrid grid = make_grid(options);
        auto splitter =
            std::make_shared<RandomKFold>(options.folds, true, options.seed);
        auto runner = std::make_shared<CNNTrialRunner>(MPI_COMM_WORLD);
        CrossValidator validator(training_dataset, splitter, runner,
                                 MPI_COMM_WORLD, true);
        const SearchResult result = validator.tune(grid);
        const CandidateResult& best = result.best();

        if (rank == 0) {
            std::filesystem::create_directories(options.results_dir);
            write_results(options.results_dir + "/fold_results.csv", result);
            write_history(options.results_dir + "/training_history.csv", result);
            std::cout << "\nBest activation candidate: " << best.config.name << '\n'
                      << "Cross-validated physical MSE: "
                      << best.mean_validation_mse << " +/- "
                      << best.validation_stddev << '\n';
        }

        // The test set is loaded only after the activation has been selected.
        Dataset test_dataset(options.test_path);
        const auto training_indices = training_dataset.all_indices();
        const auto test_indices = test_dataset.all_indices();
        const NormalizationStats normalization =
            training_dataset.fit_normalization(training_indices);

        TrialConfig final_config = best.config;
        // Trainer performs its final validation pass after the epoch loop. Set
        // the interval beyond the epoch budget so the untouched test set is
        // not observed during training.
        final_config.training.validation_interval = final_config.training.epochs + 1;
        ModelFactory factory;
        auto final_model = factory.build(
            final_config, training_dataset.sdf_height(), training_dataset.sdf_width(),
            training_dataset.scalar_features(), final_config.training.seed,
            MPI_COMM_WORLD);

        TrainingRunContext final_context;
        final_context.final_subdirectory = options.diagnostics;
        final_context.random_seed = final_config.training.seed;
        final_context.training_dataset_path = options.train_path;
        final_context.validation_dataset_path = options.test_path;
        Trainer trainer(MPI_COMM_WORLD);
        const TrainingResult final_result = trainer.fit(
            *final_model, training_dataset, training_indices, test_dataset,
            test_indices, normalization, final_config.loss, final_config.training,
            final_config.training.seed, false, final_context, &final_config);

        if (rank == 0) {
            std::cout << "Final untouched-test physical MSE: "
                      << final_result.validation_mse << '\n';
            std::ofstream summary(options.results_dir + "/summary.txt");
            if (summary) {
                summary << "Activation-function tuning\n"
                        << "Best candidate: " << best.config.name << '\n'
                        << "Cross-validation physical MSE: "
                        << best.mean_validation_mse << " +/- "
                        << best.validation_stddev << '\n'
                        << "Final untouched-test physical MSE: "
                        << final_result.validation_mse << '\n';
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "activation-function-tuning failed on rank " << rank
                  << ": " << error.what() << '\n';
        if (world_size > 1) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        status = 1;
    }

    MPI_Finalize();
    return status;
}
