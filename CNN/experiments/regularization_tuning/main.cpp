// Weight-regularization (L1/L2) tuning on the optimizer_comparison /
// layer_tuning winning topology, at the project's original lr = 1e-5.
//
// Scientific question: the earlier small {128,64} topology selected
// lambda* = 0 for both L1 and L2 at lr = 1e-5
// (~930k parameters over 1713 samples), where the measured train/validation
// gap was +0.000413 against a 0.001397 fold spread -- essentially no
// overfitting for a penalty to remove. At lr = 1e-5 lambda* = 0 is still the
// expected answer here too. What changed is the architecture: {1024,512,256,128}
// is ~8.06M parameters, about 8.7x larger, on the same 1713 samples. This is a
// legitimately different regime even if it still moves very little per step.
//
// The experiment runs two independent one-dimensional ParameterGrid searches
// in one C++ process: L2 alone and L1 alone, with lambda = 0 as the reference
// in each grid. CrossValidator::tune evaluates every candidate on the same
// fold plan and returns all fold metrics in SearchResult. No external driver
// is needed to launch one process per lambda.
//
// Every quantity that must vary per run (fold count, epochs, seed, dataset
// path, results directory, and batch size) remains a CLI argument, so the
// search can be rebuilt and launched on the cluster without source edits.
//
// See CNN/experiments/regularization_tuning/README.md.

#include "core/Loss.hpp"
#include "core/Tensor.hpp"
#include "data/Dataset.hpp"
#include "model/ModelFactory.hpp"
#include "training/Trainer.hpp"
#include "tuning/CrossValidator.hpp"
#include "tuning/SearchSpace.hpp"
#include "tuning/TrialConfig.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mpi.h>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr size_t kDefaultFolds = 5;
constexpr size_t kDefaultEpochs = 100;
constexpr size_t kDefaultGlobalBatchSize = 64;
constexpr uint64_t kDefaultSeed = 42;
constexpr float kLearningRate = 1e-5f;
constexpr float kPhysicsWeight = 0.25f;
constexpr float kLeakyAlpha = 0.05f;
constexpr size_t kEvaluationChunk = 256;

// Winning CNN topology from optimizer_comparison / activation_tuning /
// physics_weight_tuning_lr1e3: conv5x5-dense-1024-512-256-128.
ModelBlueprint make_blueprint() {
    ModelBlueprint blueprint;
    blueprint.feature_layers.push_back(Recipes::conv2d(8, 5, 5, 0));
    blueprint.feature_layers.push_back(Recipes::activation("leakyrelu", kLeakyAlpha));
    blueprint.feature_layers.push_back(Recipes::flatten());
    for (int width : {1024, 512, 256, 128}) {
        blueprint.head_layers.push_back(Recipes::dense(width));
        blueprint.head_layers.push_back(Recipes::activation("leakyrelu", kLeakyAlpha));
    }
    blueprint.head_layers.push_back(Recipes::dense(1));
    return blueprint;
}

// Fixed grids for the lr=1e-5 regularization sweep.
const std::vector<float>& l2_grid() {
    static const std::vector<float> grid = {0.0f,   1e-4f,  3.16e-4f, 1e-3f,
                                            3.16e-3f, 1e-2f, 3.16e-2f, 1e-1f};
    return grid;
}

const std::vector<float>& l1_grid() {
    static const std::vector<float> grid = {0.0f,    6.75e-7f, 2.13e-6f, 6.75e-6f,
                                            2.13e-5f, 6.75e-5f, 2.13e-4f, 6.75e-4f};
    return grid;
}

// Fold seed derivation matching the CrossValidator convention, so a fold run
// here trains from exactly the same initialization as the same fold elsewhere.
uint64_t fold_seed(uint64_t base_seed, size_t fold_index) {
    constexpr uint64_t golden_ratio = 0x9e3779b97f4a7c15ULL;
    return base_seed ^ (golden_ratio + static_cast<uint64_t>(fold_index) +
                        (base_seed << 6U) + (base_seed >> 2U));
}

bool is_weight_tensor(const LayerParameter& parameter) {
    return parameter.tensor && parameter.name == "weights";
}

// Identical formatting to the original experiment's format_lambda:
// scientific, zero decimals, and plain "0" for the reference candidate.
std::string format_lambda(float value) {
    std::ostringstream text;
    text << std::scientific << std::setprecision(0) << value;
    return value == 0.0f ? std::string("0") : text.str();
}

size_t parse_size_value(const std::string& text, const std::string& option) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument("Invalid value for " + option + ": " + text);
    }
    size_t consumed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid value for " + option + ": " + text);
    }
    if (consumed != text.size() ||
        value > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        throw std::invalid_argument("Invalid value for " + option + ": " + text);
    }
    return static_cast<size_t>(value);
}

uint64_t parse_seed_value(const std::string& text) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument("Invalid value for --seed: " + text);
    }
    size_t consumed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid value for --seed: " + text);
    }
    if (consumed != text.size()) {
        throw std::invalid_argument("Invalid value for --seed: " + text);
    }
    return static_cast<uint64_t>(value);
}

// Physical-unit data MSE over an arbitrary index set, evaluated in chunks so
// a fold never materializes as one giant tensor.
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

// MSE of the predictor that always answers with the training-fold mean.
double baseline_mse(const Dataset& dataset,
                    const std::vector<size_t>& validation,
                    const NormalizationStats& normalization) {
    double sum = 0.0;
    size_t seen = 0;
    for (size_t offset = 0; offset < validation.size(); offset += kEvaluationChunk) {
        const size_t count = std::min(kEvaluationChunk, validation.size() - offset);
        const std::span<const size_t> chunk(validation.data() + offset, count);
        const DataBatch batch = dataset.make_batch(chunk, normalization);
        auto zeros = std::make_shared<Tensor>(batch.targets->get_shape());
        sum += static_cast<double>(Loss::physical_mse(
                   zeros, batch.targets,
                   static_cast<float>(normalization.target_std))) *
               static_cast<double>(count);
        seen += count;
    }
    return seen > 0 ? sum / static_cast<double>(seen) : 0.0;
}

struct WeightSummary {
    double sum_squares = 0.0;
    double sum_absolute = 0.0;
    double change_norm = 0.0;
};

WeightSummary summarize_weights(const CNNModel& model,
                                const std::vector<std::vector<float>>& initial) {
    WeightSummary summary;
    size_t index = 0;
    for (const auto& parameter : model.parameters()) {
        const std::vector<float>& before = initial[index++];
        if (!is_weight_tensor(parameter)) {
            continue;
        }
        const float* now = parameter.tensor->get_data();
        for (size_t i = 0; i < parameter.tensor->size(); ++i) {
            summary.sum_squares += static_cast<double>(now[i]) * now[i];
            summary.sum_absolute += std::abs(static_cast<double>(now[i]));
            const double delta = static_cast<double>(now[i]) - before[i];
            summary.change_norm += delta * delta;
        }
    }
    summary.change_norm = std::sqrt(summary.change_norm);
    return summary;
}

std::vector<std::vector<float>> snapshot_parameters(const CNNModel& model) {
    std::vector<std::vector<float>> saved;
    for (const auto& parameter : model.parameters()) {
        const float* data = parameter.tensor->get_data();
        saved.emplace_back(data, data + parameter.tensor->size());
    }
    return saved;
}

struct FoldRecord {
    std::string candidate;
    float l1_weight = 0.0f;
    float l2_weight = 0.0f;
    size_t fold = 0;
    double train_mse = 0.0;
    double validation_mse = 0.0;
    double baseline_mse = 0.0;
    double l1_penalty = 0.0;
    double l2_penalty = 0.0;
    double sum_squares = 0.0;
    double change_norm = 0.0;
    size_t epochs = 0;
};

// Trains one fold and records the paired-analysis inputs: the mean-predictor
// baseline, physical-unit training MSE, reached penalty, and weight movement.
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

        ModelFactory factory;
        auto model = factory.build(config, dataset.sdf_height(),
                                   dataset.sdf_width(), dataset.scalar_features(),
                                   seed, _communicator);
        const auto initial = snapshot_parameters(*model);

        TrainingRunContext run_context;
        run_context.mode = "cross_validation";
        run_context.candidate_index = context.candidate_index;
        run_context.candidate_count = context.candidate_count;
        run_context.fold_index = fold_index;
        run_context.fold_count = context.fold_count;
        run_context.random_seed = seed;
        run_context.training_dataset_path =
            config.training.diagnostics.training_dataset_path;
        run_context.validation_dataset_path =
            config.training.diagnostics.training_dataset_path;

        const TrainingResult result = _trainer.fit(
            *model, dataset, fold.training, dataset, fold.validation,
            normalization, config.loss, config.training, seed, false,
            run_context, &config);

        FoldRecord record;
        record.candidate = config.name;
        record.l1_weight = config.loss.l1_weight;
        record.l2_weight = config.loss.l2_weight;
        record.fold = fold_index;
        record.validation_mse = result.validation_mse;
        record.epochs = config.training.epochs;
        record.train_mse =
            evaluate_physical_mse(*model, dataset, fold.training, normalization);

        auto cached = _baseline_by_fold.find(fold_index);
        if (cached == _baseline_by_fold.end()) {
            cached = _baseline_by_fold
                         .emplace(fold_index,
                                  baseline_mse(dataset, fold.validation,
                                               normalization))
                         .first;
        }
        record.baseline_mse = cached->second;

        const WeightSummary summary = summarize_weights(*model, initial);
        record.sum_squares = summary.sum_squares;
        record.change_norm = summary.change_norm;
        record.l2_penalty = config.loss.l2_weight * summary.sum_squares;
        record.l1_penalty = config.loss.l1_weight * summary.sum_absolute;
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
    Trainer _trainer;
    std::shared_ptr<std::vector<FoldRecord>> _records;
    mutable std::map<size_t, double> _baseline_by_fold;
};

void write_fold_csv(const std::string& path,
                    const std::vector<FoldRecord>& records) {
    std::ofstream csv(path);
    if (!csv) {
        throw std::runtime_error("Cannot open " + path + " for writing.");
    }
    csv << std::setprecision(10)
        << "candidate,l1_weight,l2_weight,fold,train_mse,val_mse,baseline_mse,"
           "l1_penalty,l2_penalty,sum_w2,weight_change_norm,epochs\n";
    for (const FoldRecord& record : records) {
        csv << record.candidate << ',' << record.l1_weight << ','
            << record.l2_weight << ',' << record.fold << ',' << record.train_mse
            << ',' << record.validation_mse << ',' << record.baseline_mse << ','
            << record.l1_penalty << ',' << record.l2_penalty << ','
            << record.sum_squares << ',' << record.change_norm << ','
            << record.epochs << '\n';
    }
}

void write_history_csv(const std::string& path,
                       const SearchResult& search) {
    std::ofstream csv(path);
    if (!csv) {
        throw std::runtime_error("Cannot open " + path + " for writing.");
    }
    csv << std::setprecision(10)
        << "candidate,fold,epoch,training_objective,validation_mse\n";
    for (const CandidateResult& candidate : search.candidates) {
        if (!candidate.success) {
            continue;
        }
        for (const FoldMetrics& fold : candidate.folds) {
            for (const EpochMetrics& point : fold.history) {
                csv << candidate.config.name << ',' << fold.fold << ','
                    << point.epoch << ',' << point.training_objective << ','
                    << point.validation_mse << '\n';
            }
        }
    }
}

struct ProgramOptions {
    size_t epochs = kDefaultEpochs;
    size_t folds = kDefaultFolds;
    size_t global_batch_size = kDefaultGlobalBatchSize;
    uint64_t seed = kDefaultSeed;
    size_t validation_interval = 10;
    std::string train_path = "dataset/cnn_dataset_train.npz";
    std::string results_dir =
        "results/cross_validation/regularization_tuning";
    bool diagnostics = false;
    size_t histogram_bins = 64;
    bool smoke = false;
    bool help = false;
};

void validate_options(const ProgramOptions& options) {
    if (options.epochs == 0) {
        throw std::invalid_argument("--epochs must be positive.");
    }
    if (options.folds < 2) {
        throw std::invalid_argument("--folds must be at least 2.");
    }
    if (options.global_batch_size == 0) {
        throw std::invalid_argument("--batch-size must be positive.");
    }
    if (options.validation_interval == 0) {
        throw std::invalid_argument("--validation-interval must be positive.");
    }
    if (options.validation_interval > options.epochs) {
        throw std::invalid_argument(
            "--validation-interval cannot exceed --epochs.");
    }
    if (options.histogram_bins == 0) {
        throw std::invalid_argument("--histogram-bins must be positive.");
    }
    if (options.train_path.empty()) {
        throw std::invalid_argument("--train-path cannot be empty.");
    }
    if (options.results_dir.empty()) {
        throw std::invalid_argument("--results-dir cannot be empty.");
    }
}

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

        if (arg == "--epochs") {
            options.epochs = parse_size_value(next_value(), "--epochs");
        } else if (arg == "--folds") {
            options.folds = parse_size_value(next_value(), "--folds");
        } else if (arg == "--batch-size") {
            options.global_batch_size =
                parse_size_value(next_value(), "--batch-size");
        } else if (arg == "--seed") {
            options.seed = parse_seed_value(next_value());
        } else if (arg == "--validation-interval") {
            options.validation_interval =
                parse_size_value(next_value(), "--validation-interval");
        } else if (arg == "--train-path") {
            options.train_path = next_value();
        } else if (arg == "--results-dir") {
            options.results_dir = next_value();
        } else if (arg == "--diagnostics") {
            options.diagnostics = true;
        } else if (arg == "--no-diagnostics") {
            options.diagnostics = false;
        } else if (arg == "--histogram-bins") {
            options.histogram_bins =
                parse_size_value(next_value(), "--histogram-bins");
        } else if (arg == "--smoke") {
            options.smoke = true;
            options.epochs = 2;
            options.folds = 2;
            options.validation_interval = 1;
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
        << "Regularization (L1/L2) Tuning, layer_tuning architecture (lr=1e-5)\n\n"
        << "Usage: regularization_tuning [options]\n\n"
        << "Runs the complete L1 and L2 ParameterGrid searches in one C++ process.\n\n"
        << "Options:\n"
        << "  --epochs N               Epochs per fold (default: 100)\n"
        << "  --folds N                Number of CV folds (default: 5)\n"
        << "  --batch-size N           Global batch size (default: 64)\n"
        << "  --seed N                 Split/shuffle/initialization seed (default: 42)\n"
        << "  --validation-interval N  Validation frequency (default: 10)\n"
        << "  --train-path PATH        Training NPZ (default: dataset/cnn_dataset_train.npz)\n"
        << "  --results-dir PATH       Output directory\n"
        << "  --diagnostics            Write per-epoch training diagnostics\n"
        << "  --no-diagnostics         Disable diagnostics (default)\n"
        << "  --histogram-bins N       Activation histogram bins (default: 64)\n"
        << "  --smoke                  Run two candidates per axis, 2 folds, 2 epochs,\n"
        << "                           validating every epoch\n"
        << "  --help, -h               Show this message\n\n"
        << "The topology (conv5x5-dense-1024-512-256-128), learning rate (1e-5, Adam),\n"
        << "LeakyReLU alpha (0.05), and physics weight (0.25) are fixed.\n";
}

TrialConfig make_base_config(const std::string& axis,
                             const ProgramOptions& options) {
    TrainingConfig training;
    training.epochs = options.epochs;
    training.global_batch_size = options.global_batch_size;
    training.gradient_clip = 1.0f;
    training.seed = options.seed;
    training.shuffle = true;
    training.validation_interval = options.validation_interval;
    if (options.diagnostics) {
        training.diagnostics.enabled = true;
        training.diagnostics.results_root = options.results_dir;
        training.diagnostics.experiment_name =
            "regularization_tuning";
        training.diagnostics.run_name = axis;
        training.diagnostics.histogram_bins = options.histogram_bins;
        training.diagnostics.training_dataset_path = options.train_path;
        training.diagnostics.validation_dataset_path = options.train_path;
    }

    return TrialConfig{"regularization-" + axis,
                       make_blueprint(),
                       Recipes::adam(kLearningRate),
                       LossConfig{kPhysicsWeight, 0.0f, 0.0f},
                       training,
                       {}};
}

void add_regularization_axis(ParameterGrid& grid,
                             const std::string& axis,
                             const std::vector<float>& values) {
    std::vector<NamedChoice<float>> choices;
    choices.reserve(values.size());
    for (float value : values) {
        choices.push_back(NamedChoice<float>{format_lambda(value), value});
    }

    if (axis == "l1") {
        grid.add_choice<float>(
            "l1", std::move(choices),
            [](TrialConfig& trial, const float& value) {
                trial.loss.l1_weight = value;
            });
    } else if (axis == "l2") {
        grid.add_choice<float>(
            "l2", std::move(choices),
            [](TrialConfig& trial, const float& value) {
                trial.loss.l2_weight = value;
            });
    } else {
        throw std::invalid_argument("Unknown regularization axis: " + axis);
    }
}

SearchResult run_sweep(const Dataset& dataset,
                       const ProgramOptions& options,
                       const std::string& axis,
                       const std::vector<float>& values,
                       int rank) {
    auto records = std::make_shared<std::vector<FoldRecord>>();
    ParameterGrid grid(make_base_config(axis, options));
    add_regularization_axis(grid, axis, values);

    auto splitter =
        std::make_shared<RandomKFold>(options.folds, true, options.seed);
    auto runner = std::make_shared<RecordingRunner>(MPI_COMM_WORLD, records);
    CrossValidator validator(dataset, splitter, runner, MPI_COMM_WORLD,
                             rank == 0);
    SearchResult result = validator.tune(grid);

    if (rank == 0) {
        std::filesystem::create_directories(options.results_dir);
        const std::string prefix = options.results_dir + "/sweep_" + axis;
        write_fold_csv(prefix + ".csv", *records);
        write_history_csv(options.results_dir + "/training_history_" + axis +
                              ".csv",
                          result);

        std::cout << "\n==================== " << axis
                  << " SWEEP RESULTS ====================\n";
        for (const CandidateResult& candidate : result.candidates) {
            std::cout << candidate.config.name << " : ";
            if (!candidate.success) {
                std::cout << "FAILED (" << candidate.error << ")\n";
                continue;
            }
            std::cout << "val MSE " << candidate.mean_validation_mse << " +/- "
                      << candidate.validation_stddev << '\n';
        }
        if (result.best_index < result.candidates.size()) {
            const CandidateResult& best = result.best();
            std::cout << "Best by mean validation MSE: " << best.config.name
                      << " (" << best.mean_validation_mse << ")\n";
        }
        std::cout << "NOTE: ranking by the mean is reported for continuity only.\n"
                  << "The conclusion comes from the paired analysis of "
                  << prefix << ".csv\n"
                  << "CSV written to " << prefix << ".csv\n";
    }
    return result;
}

std::vector<float> selected_grid(const std::vector<float>& full_grid,
                                 bool smoke) {
    if (!smoke) {
        return full_grid;
    }
    return {full_grid[0], full_grid[1]};
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
                      << "  REGULARIZATION TUNING, LAYER_TUNING ARCHITECTURE (lr=1e-5)\n"
                      << "========================================================\n"
                      << "Search:          L1 and L2 ParameterGrid sweeps\n"
                      << "Topology:        conv5x5-dense-1024-512-256-128\n"
                      << "Optimizer:       Adam (lr = " << kLearningRate << ", fixed)\n"
                      << "Folds:           " << options.folds << "\n"
                      << "Epochs:          " << options.epochs << "\n"
                      << "Batch size:      " << options.global_batch_size << "\n"
                      << "Seed:            " << options.seed << "\n"
                      << "Diagnostics:     "
                      << (options.diagnostics ? "on" : "off") << "\n"
                      << "MPI ranks:       " << world_size << "\n"
                      << "Results dir:     " << options.results_dir << "\n"
                      << "--------------------------------------------------------\n";
        }

        Dataset training_dataset(options.train_path);
        const std::vector<float> l1_values =
            selected_grid(l1_grid(), options.smoke);
        const std::vector<float> l2_values =
            selected_grid(l2_grid(), options.smoke);
        run_sweep(training_dataset, options, "l1", l1_values, rank);
        run_sweep(training_dataset, options, "l2", l2_values, rank);
    } catch (const std::exception& error) {
        std::cerr << "regularization_tuning failed on rank " << rank
                  << ": " << error.what() << '\n';
        if (world_size > 1) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        status = 1;
    }

    MPI_Finalize();
    return status;
}
