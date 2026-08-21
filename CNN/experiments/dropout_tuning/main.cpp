// Dropout-rate tuning experiment.
//
// Scientific question: does dropout in the dense head improve generalization
// on this problem, and at which rate? The regularization experiment measured
// a train/validation gap of +0.000413 against a 0.001397 fold spread, so
// there is little overfitting for dropout to remove; this sweep quantifies
// whether the answer is the same for a stochastic regularizer.
//
// Selection metric: sample-weighted physical-unit validation MSE, but the
// conclusion is drawn from the PAIRED difference against rate = 0, exactly
// as in the other experiments: all candidates share one fold plan and one
// seed, which is what makes the pairing valid.
//
// The dropout recipe is present in EVERY candidate, including rate = 0
// (where the layer is the identity): a recipe consumes one layer seed during
// model construction, so keeping the structure fixed means every candidate
// starts from the same initial weights and the paired differences measure
// only the masking.
//
// Everything except the rate is held fixed: architecture, learning rate,
// physics weight, L1/L2 = 0, batch size, folds, epochs, seed.
//
// See CNN/experiments/dropout_tuning/README.md.

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

constexpr size_t kFoldCount = 5;
constexpr size_t kGlobalBatchSize = 64;
constexpr uint64_t kSeed = 42;
constexpr float kLearningRate = 1e-5f;
constexpr float kPhysicsWeight = 0.25f;
constexpr size_t kEvaluationChunk = 256;
constexpr const char* kDatasetPath = "dataset/cnn_dataset_train.npz";

// Baseline topology from make_single_trial() in CNN/main.cpp, with dropout
// after each hidden activation. The recipe is present even at rate = 0 so
// every candidate consumes the same layer seeds (identical initialization).
ModelBlueprint make_blueprint(float dropout_rate) {
    ModelBlueprint blueprint;
    blueprint.feature_layers.push_back(Recipes::conv2d(8, 5, 5, 0));
    blueprint.feature_layers.push_back(Recipes::activation("leakyrelu", 0.05f));
    blueprint.feature_layers.push_back(Recipes::flatten());
    for (int width : {128, 64}) {
        blueprint.head_layers.push_back(Recipes::dense(width));
        blueprint.head_layers.push_back(Recipes::activation("leakyrelu", 0.05f));
        blueprint.head_layers.push_back(Recipes::dropout(dropout_rate));
    }
    blueprint.head_layers.push_back(Recipes::dense(1));
    return blueprint;
}

TrialConfig make_config(const std::string& name,
                        float dropout_rate,
                        size_t epochs) {
    return TrialConfig{name,
                       make_blueprint(dropout_rate),
                       Recipes::adam(kLearningRate),
                       LossConfig{kPhysicsWeight, 0.0f, 0.0f},
                       TrainingConfig{epochs, kGlobalBatchSize, 1.0f, kSeed, true},
                       {}};
}

// Replicated from the anonymous namespace of CrossValidator.cpp so this runner
// seeds each fold exactly like CNNTrialRunner does. NOTE: unlike the other
// experiments, absolute fold values here are NOT comparable with their
// baselines: ModelFactory assigns one seed per recipe, so the extra dropout
// recipes shift the seeds of the dense layers and every candidate trains from
// a different initialization realization than the shared no-dropout
// blueprint. The paired comparison across rates is what remains valid.
uint64_t fold_seed(uint64_t base_seed, size_t fold_index) {
    constexpr uint64_t golden_ratio = 0x9e3779b97f4a7c15ULL;
    return base_seed ^ (golden_ratio + static_cast<uint64_t>(fold_index) +
                        (base_seed << 6U) + (base_seed >> 2U));
}

// Physical-unit data MSE over an index set, evaluated in chunks and run
// identically on every rank (predict() is deterministic and dropout-free).
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

// MSE of the predictor that always answers with the training-fold mean, in
// physical units (see regularization_tuning for the derivation).
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

struct FoldRecord {
    std::string candidate;
    float dropout_rate = 0.0f;
    size_t fold = 0;
    double train_mse = 0.0;
    double validation_mse = 0.0;
    double baseline_mse = 0.0;
    size_t epochs = 0;
};

// Mirrors CNNTrialRunner and additionally records the paired-analysis inputs:
// the mean-predictor baseline and the training MSE, whose distance from the
// validation MSE is the overfitting gap dropout is supposed to shrink.
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
        // Same run-context wiring as CNNTrialRunner, so enabled diagnostics
        // land in the standard candidate_NNN/fold_NNN layout with gradient
        // and weight-update statistics per epoch.
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

        // The dropout rate travels in the grid's selected_parameters label;
        // parse it back out so the CSV stores a numeric column.
        float rate = 0.0f;
        const auto selected = config.selected_parameters.find("dropout-rate");
        if (selected != config.selected_parameters.end()) {
            rate = std::stof(selected->second);
        }

        FoldRecord record;
        record.candidate = config.name;
        record.dropout_rate = rate;
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
    mutable std::map<size_t, double> _baseline_by_fold;
};

void write_csv(const std::string& path, const std::vector<FoldRecord>& records) {
    std::ofstream csv(path);
    if (!csv) {
        throw std::runtime_error("Cannot open " + path);
    }
    csv << std::setprecision(10)
        << "candidate,dropout_rate,fold,train_mse,val_mse,baseline_mse,epochs\n";
    for (const FoldRecord& record : records) {
        csv << record.candidate << ',' << record.dropout_rate << ','
            << record.fold << ',' << record.train_mse << ','
            << record.validation_mse << ',' << record.baseline_mse << ','
            << record.epochs << '\n';
    }
}

std::string format_rate(float value) {
    std::ostringstream text;
    text << value;
    return text.str();
}

int run_sweep(const Dataset& dataset,
              const std::vector<float>& values,
              size_t epochs,
              const std::string& csv_path,
              int rank,
              bool diagnostics) {
    auto records = std::make_shared<std::vector<FoldRecord>>();

    TrialConfig base = make_config("drop", 0.0f, epochs);
    // Per-epoch gradient/update/activation statistics for stability analysis
    // (requested by the group); disable with --no-diagnostics.
    base.training.diagnostics.enabled = diagnostics;
    base.training.diagnostics.results_root = "results";
    base.training.diagnostics.experiment_name = "dropout_tuning";
    base.training.diagnostics.run_name = "sweep";
    base.training.diagnostics.training_dataset_path = kDatasetPath;
    base.training.diagnostics.validation_dataset_path = kDatasetPath;
    ParameterGrid grid(base);
    std::vector<NamedChoice<float>> choices;
    choices.reserve(values.size());
    for (float value : values) {
        choices.push_back(NamedChoice<float>{format_rate(value), value});
    }
    // The whole blueprint is rebuilt per candidate: the structure is shared,
    // only the rate inside the dropout recipes changes.
    grid.add_choice<float>("dropout-rate", choices,
                           [](TrialConfig& trial, const float& value) {
                               trial.model = make_blueprint(value);
                           });

    auto splitter = std::make_shared<RandomKFold>(kFoldCount, true, kSeed);
    auto runner = std::make_shared<RecordingRunner>(MPI_COMM_WORLD, records);
    CrossValidator validator(dataset, splitter, runner, MPI_COMM_WORLD, true);

    const SearchResult result = validator.tune(grid);

    if (rank == 0) {
        write_csv(csv_path, *records);
        std::cout << "\n============== DROPOUT SWEEP RESULTS ==============\n";
        for (const CandidateResult& candidate : result.candidates) {
            std::cout << candidate.config.name << " : ";
            if (!candidate.success) {
                std::cout << "FAILED (" << candidate.error << ")\n";
                continue;
            }
            std::cout << "val MSE " << candidate.mean_validation_mse << " +/- "
                      << candidate.validation_stddev << '\n';
        }
        std::cout << "Best by mean validation MSE: "
                  << result.best().config.name << " ("
                  << result.best().mean_validation_mse << ")\n"
                  << "NOTE: ranking by the mean is reported for continuity only.\n"
                  << "The conclusion comes from the paired analysis of "
                  << csv_path << ".\n"
                  << "CSV written to " << csv_path << std::endl;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int status = 0;

    try {
        const std::string mode = argc > 1 ? argv[1] : "help";
        size_t epochs = 100;
        bool diagnostics = true;
        for (int index = 2; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--no-diagnostics") {
                diagnostics = false;
            } else if (argument == "--diagnostics") {
                diagnostics = true;
            } else {
                epochs = std::stoul(argument);
            }
        }

        if (mode == "sweep") {
            Dataset dataset(kDatasetPath);
            // 0 is the exact baseline (identical initialization); 0.5 is the
            // classic strong rate that must visibly move the metrics if
            // dropout matters at all on this problem.
            status = run_sweep(dataset, {0.0f, 0.1f, 0.2f, 0.3f, 0.5f}, epochs,
                               "results/cross_validation/dropout_tuning/"
                               "sweep_dropout.csv",
                               rank, diagnostics);
        } else if (rank == 0) {
            std::cout
                << "Usage: dropout_tuning <mode> [epochs] [--no-diagnostics]\n"
                << "  sweep [epochs]   5-rate dropout sweep, 5-fold CV "
                   "(default epochs: 100)\n"
                << "  Per-epoch gradient/weight diagnostics are written to\n"
                << "  results/dropout_tuning/sweep/ by default; disable with\n"
                << "  --no-diagnostics.\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "Dropout tuning failed on rank " << rank << ": "
                  << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return status;
}
