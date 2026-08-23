// Weight-regularization (L1/L2) tuning on the optimizer_comparison /
// layer_tuning winning topology, at the project's original lr = 1e-5.
//
// Scientific question: CNN/experiments/regularization_tuning selected
// lambda* = 0 for both L1 and L2 at lr = 1e-5 on the {128,64} topology
// (~7.5e4 parameters over 1713 samples), where the measured train/validation
// gap was +0.000413 against a 0.001397 fold spread -- essentially no
// overfitting for a penalty to remove. At lr = 1e-5 lambda* = 0 is still the
// expected answer here too, for the same reason: the update-ratio mechanism
// that keeps the network near its initialization (~3.9e-5, measured in
// physics_weight_tuning_lr1e3's README on the old topology) is a property
// of the learning rate, not of the architecture. What changed is the
// architecture, not the rate: {1024,512,256,128} is ~8.06M parameters
// (recomputed from the layer shapes, not carried over from an earlier
// estimate) instead of ~7.5e4, on the same 1713 samples -- a ~100x larger
// network is a legitimately different regime even if it still moves very
// little per step, which is the actual reason to run this rather than
// assume the old conclusion transfers. See
// CNN/experiments/physics_weight_tuning_lr1e3, the one experiment in this
// family that stayed at lr = 1e-3 (it already has real results measured
// there, which is why the learning-rate question was raised at all).
//
// Same two independent one-dimensional sweeps as the original (L2 alone,
// L1 alone, lambda = 0 as the shared reference in each), same 8-point grids,
// same fold plan and seed, same CSV schema, so
// CNN/experiments/regularization_tuning/analyze.py runs against this
// sweep's output unmodified. Held fixed: architecture, learning rate,
// physics weight, LeakyReLU alpha, batch size, folds, seed -- copied
// verbatim from CNN/experiments/optimizer_comparison/main.cpp.
//
// Segmented execution: --mode cv --axis <l1|l2> --lambda V runs exactly one
// candidate across all folds and writes <results-dir>/<axis>_<label>.csv.
// The companion orchestrator.py drives both fixed grids one invocation at a
// time with an idempotent resume state, and reuses the single lambda = 0
// run for both axes (it is the same TrialConfig either way) instead of
// running it twice.
//
// Every quantity that must vary per SLURM job (fold count, epochs, seed,
// dataset path, results directory, and which candidate to run) is a CLI
// argument, not a compiled-in constant: none of this requires recompiling
// on the cluster.
//
// See CNN/experiments/regularization_tuning_bigarch/README.md.

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
#include <map>
#include <memory>
#include <mpi.h>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kDefaultFolds = 5;
constexpr size_t kDefaultEpochs = 100;
constexpr size_t kDefaultGlobalBatchSize = 64;
constexpr uint64_t kDefaultSeed = 42;
// Fixed, not a tunable axis of this experiment: the project's original
// learning rate, unchanged from CNN/experiments/regularization_tuning.
// Only the architecture changed relative to that original (see the header
// comment above for why lambda* = 0 is still the expected outcome here).
constexpr float kLearningRate = 1e-5f;
constexpr float kPhysicsWeight = 0.25f;
constexpr float kLeakyAlpha = 0.05f;
constexpr size_t kEvaluationChunk = 256;

// Winning CNN topology from optimizer_comparison / activation_tuning /
// physics_weight_tuning_lr1e3: conv5x5-dense-1024-512-256-128. Copied
// verbatim so this experiment stays comparable to those three.
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

// The fixed grids, identical to regularization_tuning, for comparability
// with the lr=1e-5 sweep. Both include 0 as their own reference row (the
// analyze.py contract requires it independently per axis), but the
// orchestrator computes that single shared run once, not twice.
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
// here trains from exactly the same initialization as the same fold
// elsewhere.
uint64_t fold_seed(uint64_t base_seed, size_t fold_index) {
    constexpr uint64_t golden_ratio = 0x9e3779b97f4a7c15ULL;
    return base_seed ^ (golden_ratio + static_cast<uint64_t>(fold_index) +
                        (base_seed << 6U) + (base_seed >> 2U));
}

bool is_weight_tensor(const LayerParameter& parameter) {
    return parameter.tensor && parameter.name == "weights";
}

// Identical formatting to the original experiment's format_lambda:
// scientific, zero decimals (so 6.75e-4 prints as "7e-04"), plain "0" for
// the reference candidate. Kept byte-identical so labels are directly
// comparable at a glance with the lr=1e-5 sweep; the exact lambda value is
// still recorded in full precision in the l1_weight/l2_weight CSV columns,
// so nothing is lost numerically.
std::string format_lambda(float value) {
    std::ostringstream text;
    text << std::scientific << std::setprecision(0) << value;
    return value == 0.0f ? std::string("0") : text.str();
}

// Physical-unit data MSE over an arbitrary index set, evaluated in chunks so
// a fold never materializes as one giant tensor. Identical to the original
// experiment's evaluate_physical_mse.
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
// Identical to the original experiment's baseline_mse.
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

// Trains one fold, supplies the diagnostics run context (unlike the
// original experiment, which had no diagnostics wiring at all), and records
// the paired-analysis inputs: the mean-predictor baseline, the physical-unit
// training MSE, the penalty actually reached, and the weight-norm evidence
// that lambda did something.
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
                                   dataset.sdf_width(),
                                   dataset.scalar_features(), seed,
                                   _communicator);
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
                       const std::string& candidate_label,
                       const CandidateResult& candidate) {
    std::ofstream csv(path);
    if (!csv) {
        throw std::runtime_error("Cannot open " + path + " for writing.");
    }
    csv << std::setprecision(10)
        << "candidate,fold,epoch,training_objective,validation_mse\n";
    for (const FoldMetrics& fold : candidate.folds) {
        for (const EpochMetrics& point : fold.history) {
            csv << candidate_label << ',' << fold.fold << ',' << point.epoch
                << ',' << point.training_objective << ','
                << point.validation_mse << '\n';
        }
    }
}

struct ProgramOptions {
    std::string mode = "cv"; // "cv" or "probe"
    std::string axis;        // "l1" or "l2", required
    bool has_lambda = false;
    float lambda = 0.0f;
    size_t epochs = kDefaultEpochs;
    size_t folds = kDefaultFolds;
    size_t probe_fold = 0;
    size_t global_batch_size = kDefaultGlobalBatchSize;
    uint64_t seed = kDefaultSeed;
    size_t validation_interval = 0; // 0 means "use the Trainer default"
    std::string train_path = "dataset/cnn_dataset_train.npz";
    std::string results_dir = "results/cross_validation/regularization_tuning_bigarch";
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
        } else if (arg == "--axis") {
            options.axis = next_value();
        } else if (arg == "--lambda") {
            options.lambda = std::stof(next_value());
            options.has_lambda = true;
        } else if (arg == "--epochs") {
            options.epochs = std::stoul(next_value());
        } else if (arg == "--folds") {
            options.folds = std::stoul(next_value());
        } else if (arg == "--probe-fold") {
            options.probe_fold = std::stoul(next_value());
        } else if (arg == "--batch-size") {
            options.global_batch_size = std::stoul(next_value());
        } else if (arg == "--seed") {
            options.seed = std::stoull(next_value());
        } else if (arg == "--validation-interval") {
            options.validation_interval = std::stoul(next_value());
        } else if (arg == "--train-path") {
            options.train_path = next_value();
        } else if (arg == "--results-dir") {
            options.results_dir = next_value();
        } else if (arg == "--diagnostics") {
            options.diagnostics = true;
        } else if (arg == "--no-diagnostics") {
            options.diagnostics = false;
        } else if (arg == "--histogram-bins") {
            options.histogram_bins = std::stoul(next_value());
        } else if (arg == "--smoke") {
            options.epochs = 2;
            options.folds = 2;
        } else if (arg == "--help" || arg == "-h") {
            options.help = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }
    if (options.mode != "cv" && options.mode != "probe") {
        throw std::invalid_argument("--mode must be cv or probe.");
    }
    if (options.axis != "l1" && options.axis != "l2") {
        throw std::invalid_argument("--axis must be l1 or l2.");
    }
    if (!options.has_lambda) {
        throw std::invalid_argument(
            "--lambda V is required (one candidate per invocation; see "
            "orchestrator.py for the segmented sweep).");
    }
    return options;
}

void print_help() {
    std::cout
        << "Regularization (L1/L2) Tuning, layer_tuning architecture (lr=1e-5)\n\n"
        << "Usage: regularization_tuning_bigarch [options]\n\n"
        << "Options:\n"
        << "  --mode <cv|probe>        cv:    5-fold CV for ONE lambda (default)\n"
        << "                           probe: one fold, long run, to pick the epoch budget\n"
        << "  --axis <l1|l2>           Required. Which penalty this invocation sets.\n"
        << "  --lambda V                Required. Penalty weight for this invocation.\n"
        << "  --epochs N               Epochs per fold (default: 100)\n"
        << "  --folds N                Number of CV folds (default: 5)\n"
        << "  --probe-fold N           Fold index used by probe mode (default: 0)\n"
        << "  --batch-size N           Global batch size (default: 64)\n"
        << "  --seed N                 Split/shuffle/initialization seed (default: 42)\n"
        << "  --validation-interval N  Validation frequency (CV default 10, probe 1)\n"
        << "  --train-path PATH        Training NPZ (default: dataset/cnn_dataset_train.npz)\n"
        << "  --results-dir PATH       Output directory\n"
        << "  --diagnostics            Write per-epoch training diagnostics\n"
        << "  --no-diagnostics         Disable diagnostics (default)\n"
        << "  --histogram-bins N       Activation histogram bins (default: 64)\n"
        << "  --smoke                  Shortcut for 2 folds, 2 epochs\n"
        << "  --help, -h               Show this message\n\n"
        << "Note: topology (conv5x5-dense-1024-512-256-128), learning rate (1e-5, Adam),\n"
        << "LeakyReLU alpha (0.05), and physics weight (0.25) are fixed, not CLI options:\n"
        << "this experiment holds everything but the regularization axis constant.\n";
}

TrialConfig make_config(const std::string& axis,
                        float lambda,
                        const ProgramOptions& options,
                        const std::string& diagnostics_run_name) {
    TrainingConfig training;
    training.epochs = options.epochs;
    training.global_batch_size = options.global_batch_size;
    training.gradient_clip = 1.0f;
    training.seed = options.seed;
    training.shuffle = true;
    if (options.validation_interval != 0) {
        training.validation_interval = options.validation_interval;
    }
    if (options.diagnostics) {
        training.diagnostics.enabled = true;
        training.diagnostics.results_root = options.results_dir;
        training.diagnostics.experiment_name = "regularization_tuning_bigarch";
        training.diagnostics.run_name = diagnostics_run_name;
        training.diagnostics.histogram_bins = options.histogram_bins;
        training.diagnostics.training_dataset_path = options.train_path;
        training.diagnostics.validation_dataset_path = options.train_path;
    }

    const float l1 = axis == "l1" ? lambda : 0.0f;
    const float l2 = axis == "l2" ? lambda : 0.0f;
    TrialConfig config{"reg [" + axis + "=" + format_lambda(lambda) + "]",
                       make_blueprint(),
                       Recipes::adam(kLearningRate),
                       LossConfig{kPhysicsWeight, l1, l2},
                       training,
                       {}};
    return config;
}

std::string candidate_label(const std::string& axis, float lambda) {
    return axis + "_" + format_lambda(lambda);
}

// One (axis, lambda) candidate, K folds. Writes
// <results-dir>/<axis>_<label>.csv plus a per-fold epoch history.
CandidateResult run_candidate(const Dataset& dataset,
                              const ProgramOptions& options,
                              int rank) {
    const std::string label = candidate_label(options.axis, options.lambda);
    const TrialConfig config =
        make_config(options.axis, options.lambda, options, label);

    auto splitter =
        std::make_shared<RandomKFold>(options.folds, true, options.seed);
    auto records = std::make_shared<std::vector<FoldRecord>>();
    auto runner = std::make_shared<RecordingRunner>(MPI_COMM_WORLD, records);
    CrossValidator validator(dataset, splitter, runner, MPI_COMM_WORLD,
                             rank == 0);
    const CandidateResult result = validator.evaluate(config);

    if (rank == 0) {
        std::filesystem::create_directories(options.results_dir);
        write_fold_csv(options.results_dir + "/" + label + ".csv", *records);
        write_history_csv(options.results_dir + "/" + label + "_history.csv",
                          label, result);
        std::cout << "[" << label << "] mean validation physical MSE: "
                  << std::setprecision(9) << result.mean_validation_mse
                  << " +/- " << result.validation_stddev << std::endl;
    }
    return result;
}

// One fold, long run, validating every epoch. Used to choose the epoch
// budget at lr=1e-5 for this experiment specifically: the caller reads the
// per-epoch history to see where validation MSE bottoms out and whether it
// turns back up -- now a live question, since this topology has enough
// capacity to actually overfit.
int run_probe(const Dataset& dataset,
              const ProgramOptions& options,
              int rank) {
    ProgramOptions probe_options = options;
    if (probe_options.validation_interval == 0) {
        probe_options.validation_interval = 1;
    }
    const std::string label = candidate_label(options.axis, options.lambda);
    const TrialConfig config =
        make_config(options.axis, options.lambda, probe_options,
                   "probe_" + label);

    RandomKFold splitter(options.folds, true, options.seed);
    const auto folds = splitter.split(dataset.num_samples());
    if (options.probe_fold >= folds.size()) {
        throw std::invalid_argument("--probe-fold is out of range.");
    }
    const FoldIndices& fold = folds[options.probe_fold];

    const uint64_t seed = fold_seed(config.training.seed, options.probe_fold);
    const NormalizationStats normalization =
        dataset.fit_normalization(fold.training);

    ModelFactory factory;
    auto model = factory.build(config, dataset.sdf_height(),
                               dataset.sdf_width(), dataset.scalar_features(),
                               seed, MPI_COMM_WORLD);

    TrainingRunContext run_context;
    run_context.mode = "cross_validation";
    run_context.candidate_index = 0;
    run_context.candidate_count = 1;
    run_context.fold_index = options.probe_fold;
    run_context.fold_count = folds.size();
    run_context.random_seed = seed;
    run_context.training_dataset_path = options.train_path;
    run_context.validation_dataset_path = options.train_path;

    Trainer trainer(MPI_COMM_WORLD);
    const TrainingResult result =
        trainer.fit(*model, dataset, fold.training, dataset, fold.validation,
                    normalization, config.loss, config.training, seed,
                    rank == 0, run_context, &config);

    if (rank == 0) {
        std::filesystem::create_directories(options.results_dir);
        const std::string path =
            options.results_dir + "/probe_" + label + ".csv";
        std::ofstream csv(path);
        if (!csv) {
            throw std::runtime_error("Cannot open " + path + " for writing.");
        }
        csv << std::setprecision(10)
            << "candidate,fold,epoch,training_objective,validation_mse\n";
        for (const EpochMetrics& point : result.history) {
            csv << label << ',' << options.probe_fold << ',' << point.epoch
                << ',' << point.training_objective << ','
                << point.validation_mse << '\n';
        }
        const double baseline =
            baseline_mse(dataset, fold.validation, normalization);
        std::cout << "\nProbe finished: " << label << ", fold "
                  << options.probe_fold << ", " << options.epochs
                  << " epochs.\n"
                  << "  final validation physical MSE: "
                  << std::setprecision(9) << result.validation_mse << '\n'
                  << "  mean-predictor baseline MSE:   " << baseline << '\n'
                  << "  per-epoch history written to:  " << path << std::endl;
    }
    return 0;
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

        if (rank == 0) {
            std::cout << "========================================================\n"
                      << "  REGULARIZATION TUNING, LAYER_TUNING ARCHITECTURE (lr=1e-5)   \n"
                      << "========================================================\n"
                      << "Mode:            " << options.mode << "\n"
                      << "Axis:            " << options.axis << "\n"
                      << "Lambda:          " << options.lambda << "\n"
                      << "Topology:        conv5x5-dense-1024-512-256-128\n"
                      << "Optimizer:       Adam (lr = " << kLearningRate << ", fixed)\n"
                      << "Folds:           " << options.folds << "\n"
                      << "Epochs:          " << options.epochs << "\n"
                      << "Batch size:      " << options.global_batch_size << "\n"
                      << "Seed:            " << options.seed << "\n"
                      << "Diagnostics:     " << (options.diagnostics ? "on" : "off") << "\n"
                      << "MPI ranks:       " << world_size << "\n"
                      << "Results dir:     " << options.results_dir << "\n"
                      << "--------------------------------------------------------\n";
        }

        Dataset training_dataset(options.train_path);

        if (options.mode == "probe") {
            status = run_probe(training_dataset, options, rank);
        } else {
            run_candidate(training_dataset, options, rank);
        }
    } catch (const std::exception& error) {
        std::cerr << "regularization_tuning_bigarch failed on rank " << rank
                  << ": " << error.what() << '\n';
        if (world_size > 1) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        status = 1;
    }

    MPI_Finalize();
    return status;
}
