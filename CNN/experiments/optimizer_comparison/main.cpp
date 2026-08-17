// Optimizer Comparison Experiment
//
// Scientific question: How do different optimization algorithms (vanilla SGD,
// SGD with Momentum, AdaGrad, RMSprop, and Adam) perform when training the CNN
// surrogate model on the airfoil ice-acceleration dataset under the SIMM
// physics-informed loss?
//
// Follows the cross-validation guidelines documented in docs/cross_validation.md:
// - Deterministic K-fold split using RandomKFold on the training dataset.
// - All candidates are evaluated on the exact same folds with identical seeds.
// - Selection metric: sample-weighted physical-unit validation MSE.
// - Untouched test set (cnn_dataset_test.npz) is evaluated only once after
//   model selection by refitting the winning optimizer configuration on the
//   entire training set.
//
// Architecture and loss weights are held constant across candidates:
// - Feature extractor: Conv2D(out=8, kernel=5, stride=5) -> LeakyReLU(0.05) -> Flatten
// - Fusion: concatenate scalar inputs (Reynolds number, AoA)
// - Regression head: Dense(1024) -> LeakyReLU -> Dense(512) -> LeakyReLU -> Dense(256) -> LeakyReLU -> Dense(128) -> LeakyReLU -> Dense(1)
// - Physics loss: SIMM residual weight = 0.25

#include "core/Loss.hpp"
#include "data/Dataset.hpp"
#include "model/ModelFactory.hpp"
#include "training/Trainer.hpp"
#include "tuning/CrossValidator.hpp"
#include "tuning/SearchSpace.hpp"
#include "tuning/TrialConfig.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mpi.h>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr size_t kDefaultFolds = 5;
constexpr size_t kDefaultEpochs = 100;
constexpr size_t kDefaultGlobalBatchSize = 64;
constexpr uint64_t kDefaultSeed = 42;
constexpr float kDefaultPhysicsWeight = 0.25f;
constexpr size_t kEvaluationChunk = 256;

// Winning CNN topology from topology search: conv5x5-dense-1024-512-256-128
ModelBlueprint make_blueprint(float leaky_alpha = 0.05f) {
    ModelBlueprint blueprint;
    blueprint.feature_layers.push_back(Recipes::conv2d(8, 5, 5, 0));
    blueprint.feature_layers.push_back(Recipes::activation("leakyrelu", leaky_alpha));
    blueprint.feature_layers.push_back(Recipes::flatten());
    for (int width : {1024, 512, 256, 128}) {
        blueprint.head_layers.push_back(Recipes::dense(width));
        blueprint.head_layers.push_back(Recipes::activation("leakyrelu", leaky_alpha));
    }
    blueprint.head_layers.push_back(Recipes::dense(1));
    return blueprint;
}

// Fold seed derivation matching CrossValidator conventions
uint64_t fold_seed(uint64_t base_seed, size_t fold_index) {
    constexpr uint64_t golden_ratio = 0x9e3779b97f4a7c15ULL;
    return base_seed ^ (golden_ratio + static_cast<uint64_t>(fold_index) +
                        (base_seed << 6U) + (base_seed >> 2U));
}

// Physical-unit MSE evaluation over an index set
double evaluate_physical_mse(CNNModel& model,
                             const Dataset& dataset,
                             const std::vector<size_t>& indices,
                             const NormalizationStats& normalization) {
    double sum = 0.0;
    size_t seen = 0;
    for (size_t offset = 0; offset < indices.size(); offset += kEvaluationChunk) {
        const size_t count = std::min(kEvaluationChunk, indices.size() - offset);
        const std::span<const size_t> chunk(indices.data() + offset, count);
        const DataBatch batch = dataset.make_batch(chunk, normalization);
        const auto predictions = model.predict(batch.sdf, batch.scalars);
        sum += static_cast<double>(Loss::physical_mse(
                   predictions, batch.targets,
                   static_cast<float>(normalization.target_std))) *
               static_cast<double>(count);
        seen += count;
    }
    return seen > 0 ? sum / static_cast<double>(seen) : 0.0;
}

struct FoldRecord {
    std::string candidate;
    std::string optimizer_name;
    size_t fold = 0;
    double train_mse = 0.0;
    double validation_mse = 0.0;
    double training_objective = 0.0;
    size_t epochs = 0;
};

// Custom recording runner to capture detailed fold metrics
class RecordingRunner : public TrialRunner {
public:
    RecordingRunner(MPI_Comm communicator,
                    std::shared_ptr<std::vector<FoldRecord>> records)
        : _communicator(communicator),
          _trainer(communicator),
          _records(std::move(records)) {}

    FoldMetrics run(const TrialConfig& config,
                    const Dataset& dataset,
                    const FoldIndices& fold,
                    size_t fold_index) const override {
        return run(config, dataset, fold, fold_index,
                   Context{0, 1, fold_index + 1});
    }

    FoldMetrics run(const TrialConfig& config,
                    const Dataset& dataset,
                    const FoldIndices& fold,
                    size_t fold_index,
                    const Context& context) const override {
        const uint64_t seed = fold_seed(config.training.seed, fold_index);
        const NormalizationStats normalization =
            dataset.fit_normalization(fold.training);

        auto model = _model_factory.build(config, dataset.sdf_height(),
                                          dataset.sdf_width(),
                                          dataset.scalar_features(), seed,
                                          _communicator);

        TrainingRunContext run_context;
        run_context.mode = "cross_validation";
        run_context.candidate_index = context.candidate_index;
        run_context.candidate_count = context.candidate_count;
        run_context.fold_index = fold_index;
        run_context.fold_count = context.fold_count;
        run_context.random_seed = seed;
        run_context.training_dataset_path =
            config.training.diagnostics.training_dataset_path;
        // CV validation samples come from the same training dataset.
        run_context.validation_dataset_path =
            config.training.diagnostics.training_dataset_path;

        const TrainingResult result = _trainer.fit(
            *model, dataset, fold.training, dataset, fold.validation,
            normalization, config.loss, config.training, seed, false,
            run_context, &config);

        FoldRecord record;
        record.candidate = config.name;
        record.optimizer_name = config.optimizer.description();
        record.fold = fold_index;
        record.validation_mse = result.validation_mse;
        record.training_objective = result.training_objective;
        record.epochs = config.training.epochs;
        record.train_mse =
            evaluate_physical_mse(*model, dataset, fold.training, normalization);

        _records->push_back(record);

        return FoldMetrics{fold_index,
                           result.training_objective,
                           result.validation_mse,
                           result.training_samples,
                           result.validation_samples,
                           result.history};
    }

private:
    MPI_Comm _communicator;
    ModelFactory _model_factory;
    Trainer _trainer;
    std::shared_ptr<std::vector<FoldRecord>> _records;
};

void write_fold_records_csv(const std::string& path,
                            const std::vector<FoldRecord>& records) {
    std::ofstream csv(path);
    if (!csv) {
        throw std::runtime_error("Cannot open " + path + " for writing.");
    }
    csv << std::setprecision(10)
        << "candidate,optimizer,fold,train_mse,validation_mse,training_objective,epochs\n";
    for (const auto& r : records) {
        csv << '"' << r.candidate << "\",\"" << r.optimizer_name << "\","
            << r.fold << ',' << r.train_mse << ',' << r.validation_mse << ','
            << r.training_objective << ',' << r.epochs << '\n';
    }
}

void write_history_csv(const std::string& path, const SearchResult& result) {
    std::ofstream csv(path);
    if (!csv) {
        throw std::runtime_error("Cannot open " + path + " for writing.");
    }
    csv << std::setprecision(10)
        << "candidate,fold,epoch,training_objective,validation_mse\n";
    for (const auto& candidate : result.candidates) {
        if (!candidate.success) continue;
        for (const auto& fold : candidate.folds) {
            for (const auto& point : fold.history) {
                csv << '"' << candidate.config.name << "\","
                    << fold.fold << ',' << point.epoch << ','
                    << point.training_objective << ',' << point.validation_mse << '\n';
            }
        }
    }
}

struct ProgramOptions {
    std::string mode = "compare"; // "compare", "grid", "smoke"
    size_t epochs = kDefaultEpochs;
    size_t folds = kDefaultFolds;
    size_t global_batch_size = kDefaultGlobalBatchSize;
    uint64_t seed = kDefaultSeed;
    float physics_weight = kDefaultPhysicsWeight;
    std::string train_path = "dataset/cnn_dataset_train.npz";
    std::string test_path = "dataset/cnn_dataset_test.npz";
    std::string results_dir = "results/cross_validation/optimizer_comparison";
    bool diagnostics = false;
    size_t histogram_bins = 64;
    bool help = false;
};

ProgramOptions parse_options(int argc, char** argv) {
    ProgramOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + arg);
            }
            return argv[++i];
        };

        if (arg == "--mode") {
            options.mode = next_value();
        } else if (arg == "--epochs") {
            options.epochs = std::stoul(next_value());
        } else if (arg == "--folds") {
            options.folds = std::stoul(next_value());
        } else if (arg == "--batch-size") {
            options.global_batch_size = std::stoul(next_value());
        } else if (arg == "--seed") {
            options.seed = std::stoull(next_value());
        } else if (arg == "--physics-weight") {
            options.physics_weight = std::stof(next_value());
        } else if (arg == "--train-path") {
            options.train_path = next_value();
        } else if (arg == "--test-path") {
            options.test_path = next_value();
        } else if (arg == "--results-dir") {
            options.results_dir = next_value();
        } else if (arg == "--diagnostics") {
            options.diagnostics = true;
        } else if (arg == "--no-diagnostics") {
            options.diagnostics = false;
        } else if (arg == "--histogram-bins") {
            options.histogram_bins = std::stoul(next_value());
        } else if (arg == "--smoke") {
            options.mode = "smoke";
            options.epochs = 2;
            options.folds = 2;
        } else if (arg == "--help" || arg == "-h") {
            options.help = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }
    return options;
}

void print_help() {
    std::cout
        << "Optimizer Comparison Experiment for CNN Wing Ice Model\n\n"
        << "Usage: optimizer_comparison [options]\n\n"
        << "Options:\n"
        << "  --mode <compare|grid|smoke> Comparison mode (default: compare)\n"
        << "                                compare: compares 5 optimizers at standard tuned LRs\n"
        << "                                grid:    evaluates 5 optimizers x multiple LRs\n"
        << "                                smoke:   fast 2-fold 2-epoch sanity run\n"
        << "  --epochs <N>                Training epochs per fold (default: 100)\n"
        << "  --folds <N>                 Number of CV folds (default: 5)\n"
        << "  --batch-size <N>            Global batch size (default: 64)\n"
        << "  --seed <N>                  Random seed (default: 42)\n"
        << "  --physics-weight <V>        SIMM loss weight (default: 0.25)\n"
        << "  --train-path <PATH>         Path to training NPZ (default: dataset/cnn_dataset_train.npz)\n"
        << "  --test-path <PATH>          Path to test NPZ (default: dataset/cnn_dataset_test.npz)\n"
        << "  --results-dir <PATH>        Output directory for CSVs and logs\n"
        << "  --diagnostics               Enable pre/post activation histograms and gradient diagnostics\n"
        << "  --histogram-bins <N>        Fixed histogram bins for activations (default: 64)\n"
        << "  --smoke                     Shortcut for --mode smoke --epochs 2 --folds 2\n"
        << "  --help, -h                  Display this help message\n";
}

TrialConfig make_base_config(const ProgramOptions& options) {
    TrainingConfig training_cfg{options.epochs, options.global_batch_size, 1.0f, options.seed, true};
    if (options.diagnostics) {
        training_cfg.diagnostics.enabled = true;
        training_cfg.diagnostics.results_root = options.results_dir;
        training_cfg.diagnostics.experiment_name = "optimizer_comparison";
        training_cfg.diagnostics.histogram_bins = options.histogram_bins;
        training_cfg.diagnostics.training_dataset_path = options.train_path;
        training_cfg.diagnostics.validation_dataset_path = options.test_path;
    }

    return TrialConfig{
        "optimizer-trial",
        make_blueprint(0.05f),
        Recipes::adam(), // default placeholder replaced by grid
        LossConfig{options.physics_weight, 0.0f, 0.0f},
        training_cfg,
        {}};
}

ParameterGrid build_comparison_grid(const ProgramOptions& options) {
    TrialConfig base = make_base_config(options);
    ParameterGrid grid(base);

    // Standard comparison across all 5 optimizers using their class default parameters (lr = 1e-3):
    // 1. SGD (lr = 1e-3)
    // 2. SGD with Momentum (lr = 1e-3, momentum = 0.9)
    // 3. AdaGrad (lr = 1e-3, eps = 1e-8)
    // 4. RMSprop (lr = 1e-3, decay = 0.9, eps = 1e-8)
    // 5. Adam (lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8)
    std::vector<NamedChoice<OptimizerRecipe>> choices = {
        {"SGD (lr=1e-3)", Recipes::sgd()},
        {"SGD+Momentum (lr=1e-3, m=0.9)", Recipes::sgd_momentum()},
        {"AdaGrad (lr=1e-3)", Recipes::adagrad()},
        {"RMSprop (lr=1e-3)", Recipes::rmsprop()},
        {"Adam (lr=1e-3)", Recipes::adam()}
    };

    grid.add_choice<OptimizerRecipe>(
        "optimizer", choices,
        [](TrialConfig& trial, const OptimizerRecipe& opt) {
            trial.optimizer = opt;
        });

    return grid;
}

ParameterGrid build_sweep_grid(const ProgramOptions& options) {
    TrialConfig base = make_base_config(options);
    ParameterGrid grid(base);

    // Sweep: all 5 optimizers over learning rate scales
    std::vector<NamedChoice<OptimizerRecipe>> choices = {
        // SGD
        {"SGD_1e-4", Recipes::sgd(1e-4f, 0.0f)},
        {"SGD_1e-3", Recipes::sgd(1e-3f, 0.0f)},
        {"SGD_1e-2", Recipes::sgd(1e-2f, 0.0f)},
        // SGD with Momentum
        {"SGDM_1e-4", Recipes::sgd_momentum(1e-4f, 0.9f)},
        {"SGDM_1e-3", Recipes::sgd_momentum(1e-3f, 0.9f)},
        {"SGDM_1e-2", Recipes::sgd_momentum(1e-2f, 0.9f)},
        // AdaGrad
        {"AdaGrad_1e-3", Recipes::adagrad(1e-3f)},
        {"AdaGrad_1e-2", Recipes::adagrad(1e-2f)},
        {"AdaGrad_1e-1", Recipes::adagrad(1e-1f)},
        // RMSprop
        {"RMSprop_1e-5", Recipes::rmsprop(1e-5f)},
        {"RMSprop_1e-4", Recipes::rmsprop(1e-4f)},
        {"RMSprop_1e-3", Recipes::rmsprop(1e-3f)},
        // Adam
        {"Adam_1e-5", Recipes::adam(1e-5f)},
        {"Adam_1e-4", Recipes::adam(1e-4f)},
        {"Adam_1e-3", Recipes::adam(1e-3f)},
    };

    grid.add_choice<OptimizerRecipe>(
        "optimizer_candidate", choices,
        [](TrialConfig& trial, const OptimizerRecipe& opt) {
            trial.optimizer = opt;
        });

    return grid;
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
        ProgramOptions options = parse_options(argc, argv);
        if (options.help) {
            if (rank == 0) print_help();
            MPI_Finalize();
            return 0;
        }

        if (rank == 0) {
            std::cout << "========================================================\n"
                      << "       CNN OPTIMIZER COMPARISON EXPERIMENT              \n"
                      << "========================================================\n"
                      << "Mode:            " << options.mode << "\n"
                      << "Folds:           " << options.folds << "\n"
                      << "Epochs:          " << options.epochs << "\n"
                      << "Batch size:      " << options.global_batch_size << "\n"
                      << "Seed:            " << options.seed << "\n"
                      << "Physics weight:  " << options.physics_weight << "\n"
                      << "MPI ranks:       " << world_size << "\n"
                      << "Training dataset:" << options.train_path << "\n"
                      << "Test dataset:    " << options.test_path << "\n"
                      << "Results dir:     " << options.results_dir << "\n"
                      << "--------------------------------------------------------\n";
        }

        // Load training dataset for CV
        Dataset training_dataset(options.train_path);

        ParameterGrid grid = (options.mode == "grid")
                                 ? build_sweep_grid(options)
                                 : build_comparison_grid(options);

        auto splitter = std::make_shared<RandomKFold>(options.folds, true, options.seed);
        auto records = std::make_shared<std::vector<FoldRecord>>();
        auto runner = std::make_shared<RecordingRunner>(MPI_COMM_WORLD, records);

        CrossValidator validator(training_dataset, splitter, runner,
                                 MPI_COMM_WORLD, true);

        if (rank == 0) {
            std::cout << "Starting Cross-Validation across "
                      << grid.candidates().size() << " optimizer candidates...\n\n";
        }

        const SearchResult result = validator.tune(grid);
        const CandidateResult& best = result.best();

        if (rank == 0) {
            std::filesystem::create_directories(options.results_dir);
            const std::string folds_csv_path = options.results_dir + "/fold_results.csv";
            const std::string history_csv_path = options.results_dir + "/training_history.csv";

            write_fold_records_csv(folds_csv_path, *records);
            write_history_csv(history_csv_path, result);

            // Collect and sort successful candidates by mean validation MSE
            std::vector<CandidateResult> ranked_candidates;
            for (const auto& candidate : result.candidates) {
                if (candidate.success) {
                    ranked_candidates.push_back(candidate);
                }
            }
            std::sort(ranked_candidates.begin(), ranked_candidates.end(),
                      [](const CandidateResult& a, const CandidateResult& b) {
                          return a.mean_validation_mse < b.mean_validation_mse;
                      });

            std::cout << "\n========================================================\n"
                      << "                 CROSS-VALIDATION RESULTS               \n"
                      << "========================================================\n";
            std::cout << std::left << std::setw(35) << "Candidate / Optimizer"
                      << std::setw(20) << "Mean Val MSE"
                      << std::setw(15) << "Val StdDev"
                      << "Status\n";
            std::cout << std::string(75, '-') << "\n";

            for (const auto& candidate : result.candidates) {
                std::cout << std::left << std::setw(35) << candidate.config.name;
                if (candidate.success) {
                    std::cout << std::setw(20) << candidate.mean_validation_mse
                              << std::setw(15) << candidate.validation_stddev
                              << "SUCCESS\n";
                } else {
                    std::cout << std::setw(20) << "N/A"
                              << std::setw(15) << "N/A"
                              << "FAILED: " << candidate.error << "\n";
                }
            }

            std::cout << "\n--------------------------------------------------------\n"
                      << "TOP 3 BEST OPTIMIZERS FOR SUBSEQUENT GRID SEARCH:\n"
                      << "--------------------------------------------------------\n";
            const size_t top_count = std::min(size_t{3}, ranked_candidates.size());
            std::ofstream top3_file(options.results_dir + "/top3_optimizers.txt");
            if (top3_file) {
                top3_file << "RANK,CANDIDATE,MEAN_VAL_MSE,VAL_STDDEV\n";
            }
            for (size_t i = 0; i < top_count; ++i) {
                const auto& c = ranked_candidates[i];
                std::cout << "  #" << (i + 1) << " " << std::left << std::setw(32) << c.config.name
                          << " | Val MSE: " << c.mean_validation_mse
                          << " +/- " << c.validation_stddev << "\n";
                if (top3_file) {
                    top3_file << (i + 1) << ",\"" << c.config.name << "\","
                              << c.mean_validation_mse << "," << c.validation_stddev << "\n";
                }
            }

            std::cout << "\n--------------------------------------------------------\n"
                      << "WINNING CANDIDATE: " << best.config.name << "\n"
                      << "Mean Validation Physical MSE: " << best.mean_validation_mse
                      << " +/- " << best.validation_stddev << "\n"
                      << "Detailed fold records written to:   " << folds_csv_path << "\n"
                      << "Epoch checkpoint history saved to: " << history_csv_path << "\n"
                      << "Top 3 optimizers recorded to:      " << options.results_dir << "/top3_optimizers.txt\n"
                      << "========================================================\n\n";

            std::cout << "Refitting winning optimizer on full training dataset...\n";
        }

        // =====================================================================
        // UNTOUCHED TEST SET EVALUATION
        // As documented in docs/cross_validation.md:
        // Cross-validation uses only the training NPZ. cnn_dataset_test.npz
        // remains untouched until the winning candidate is rebuilt and trained
        // on the complete training dataset.
        // =====================================================================
        Dataset test_dataset(options.test_path);
        const auto training_indices = training_dataset.all_indices();
        const auto test_indices = test_dataset.all_indices();
        const NormalizationStats normalization =
            training_dataset.fit_normalization(training_indices);

        ModelFactory factory;
        TrialConfig final_config = best.config;
        final_config.training.validation_interval = 1;
        auto final_model = factory.build(
            final_config,
            training_dataset.sdf_height(),
            training_dataset.sdf_width(),
            training_dataset.scalar_features(),
            final_config.training.seed,
            MPI_COMM_WORLD);

        Trainer trainer(MPI_COMM_WORLD);
        TrainingRunContext final_context;
        final_context.final_subdirectory = options.diagnostics;
        final_context.random_seed = final_config.training.seed;
        final_context.training_dataset_path = options.train_path;
        final_context.validation_dataset_path = options.test_path;

        const TrainingResult final_result = trainer.fit(
            *final_model,
            training_dataset,
            training_indices,
            test_dataset,
            test_indices,
            normalization,
            final_config.loss,
            final_config.training,
            final_config.training.seed,
            (rank == 0),
            final_context,
            &final_config);

        if (rank == 0) {
            std::cout << "\n========================================================\n"
                      << "             FINAL UNBIASED TEST EVALUATION             \n"
                      << "========================================================\n"
                      << "Selected Optimizer:       " << best.config.name << "\n"
                      << "Training Objective (SIMM): " << final_result.training_objective << "\n"
                      << "Untouched Test Physical MSE: " << final_result.validation_mse << "\n"
                      << "========================================================\n";

            // Save summary file
            std::ofstream summary(options.results_dir + "/summary.txt");
            if (summary) {
                summary << "OPTIMIZER COMPARISON EXPERIMENT SUMMARY\n"
                        << "========================================\n"
                        << "Mode:                        " << options.mode << "\n"
                        << "Folds:                       " << options.folds << "\n"
                        << "Epochs:                      " << options.epochs << "\n"
                        << "Global Batch Size:           " << options.global_batch_size << "\n"
                        << "Seed:                        " << options.seed << "\n"
                        << "Winning Optimizer:           " << best.config.name << "\n"
                        << "Cross-Validation MSE:        " << best.mean_validation_mse << " +/- " << best.validation_stddev << "\n"
                        << "Final Untouched Test MSE:    " << final_result.validation_mse << "\n";
            }
        }

    } catch (const std::exception& error) {
        std::cerr << "Optimizer comparison failed on rank " << rank << ": "
                  << error.what() << '\n';
        if (world_size > 1) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        status = 1;
    }

    MPI_Finalize();
    return status;
}
