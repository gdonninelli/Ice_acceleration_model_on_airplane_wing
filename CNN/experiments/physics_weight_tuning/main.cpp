// Physics-weight (SIMM lambda) tuning experiment.
//
// Scientific question: how strongly should the physical prior C_l = 2*pi*alpha
// be enforced? The SIMM loss adds lambda * MSE(pred, standardized 2*pi*alpha)
// for samples with |alpha| <= 10 degrees. lambda = 0 is the pure data-driven
// reference, so this sweep also measures whether the physics term helps at
// all; lambda = 0.25 is the current production default.
//
// Selection metric: sample-weighted physical-unit validation MSE, but the
// conclusion is drawn from the PAIRED difference against lambda = 0, exactly
// as in the activation and regularization experiments: all candidates share
// one fold plan and one seed, which is what makes the pairing valid.
//
// The recorder additionally splits the validation MSE into the small-angle
// population (|alpha| <= 10 deg, where the prior is active) and the
// past-stall population (|alpha| > 10 deg), because the physics term can
// only be expected to act inside the mask.
//
// Everything except lambda is held fixed: architecture, learning rate,
// L1/L2 = 0, batch size, folds, epochs, seed.
//
// See CNN/experiments/physics_weight_tuning/README.md.

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
constexpr size_t kEvaluationChunk = 256;
constexpr const char* kDatasetPath = "dataset/cnn_dataset_train.npz";
// Same constant as Loss.cpp: the physics prior is masked to |alpha| <= 10 deg.
constexpr float kMaskLimitRad = 0.174533f;

// Baseline topology, identical to make_single_trial() in CNN/main.cpp.
ModelBlueprint make_blueprint() {
    ModelBlueprint blueprint;
    blueprint.feature_layers.push_back(Recipes::conv2d(8, 5, 5, 0));
    blueprint.feature_layers.push_back(Recipes::activation("leakyrelu", 0.05f));
    blueprint.feature_layers.push_back(Recipes::flatten());
    for (int width : {128, 64}) {
        blueprint.head_layers.push_back(Recipes::dense(width));
        blueprint.head_layers.push_back(Recipes::activation("leakyrelu", 0.05f));
    }
    blueprint.head_layers.push_back(Recipes::dense(1));
    return blueprint;
}

TrialConfig make_config(const std::string& name,
                        float physics_weight,
                        size_t epochs) {
    return TrialConfig{name,
                       make_blueprint(),
                       Recipes::adam(kLearningRate),
                       LossConfig{physics_weight, 0.0f, 0.0f},
                       TrainingConfig{epochs, kGlobalBatchSize, 1.0f, kSeed, true},
                       {}};
}

// Replicated from the anonymous namespace of CrossValidator.cpp so this runner
// seeds each fold exactly like CNNTrialRunner does and the numbers stay
// comparable with the other experiments.
uint64_t fold_seed(uint64_t base_seed, size_t fold_index) {
    constexpr uint64_t golden_ratio = 0x9e3779b97f4a7c15ULL;
    return base_seed ^ (golden_ratio + static_cast<uint64_t>(fold_index) +
                        (base_seed << 6U) + (base_seed >> 2U));
}

struct MaskedMse {
    double small_angle_mse = 0.0;
    size_t small_angle_count = 0;
    double large_angle_mse = 0.0;
    size_t large_angle_count = 0;
    double full_mse = 0.0;
};

// Physical-unit MSE over an index set, split by the physics mask. Evaluated in
// chunks so a fold never materializes as one giant tensor, and run identically
// on every rank so no reduction is needed.
MaskedMse evaluate_masked_mse(CNNModel& model,
                              const Dataset& dataset,
                              const std::vector<size_t>& indices,
                              const NormalizationStats& normalization) {
    const double target_std = normalization.target_std;
    double small_sum = 0.0;
    double large_sum = 0.0;
    size_t small_count = 0;
    size_t large_count = 0;

    for (size_t offset = 0; offset < indices.size(); offset += kEvaluationChunk) {
        const size_t count = std::min(kEvaluationChunk, indices.size() - offset);
        const std::span<const size_t> chunk(indices.data() + offset, count);
        const DataBatch batch = dataset.make_batch(chunk, normalization);
        const auto predictions = model.predict(batch.sdf, batch.scalars);
        const float* pred_ptr = predictions->get_data();
        const float* target_ptr = batch.targets->get_data();
        const float* alpha_ptr = batch.alpha_radians->get_data();
        for (size_t i = 0; i < count; ++i) {
            const double difference =
                static_cast<double>(pred_ptr[i] - target_ptr[i]) * target_std;
            const double squared = difference * difference;
            if (std::abs(alpha_ptr[i]) <= kMaskLimitRad) {
                small_sum += squared;
                ++small_count;
            } else {
                large_sum += squared;
                ++large_count;
            }
        }
    }

    MaskedMse result;
    result.small_angle_count = small_count;
    result.large_angle_count = large_count;
    result.small_angle_mse =
        small_count > 0 ? small_sum / static_cast<double>(small_count) : 0.0;
    result.large_angle_mse =
        large_count > 0 ? large_sum / static_cast<double>(large_count) : 0.0;
    const size_t total = small_count + large_count;
    result.full_mse =
        total > 0 ? (small_sum + large_sum) / static_cast<double>(total) : 0.0;
    return result;
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
    float physics_weight = 0.0f;
    size_t fold = 0;
    double train_mse = 0.0;
    double validation_mse = 0.0;
    double baseline_mse = 0.0;
    double val_mse_small_angle = 0.0;
    size_t small_angle_count = 0;
    double val_mse_large_angle = 0.0;
    size_t large_angle_count = 0;
    size_t epochs = 0;
};

// Mirrors CNNTrialRunner and additionally records the paired-analysis inputs:
// the mean-predictor baseline, the training MSE, and the mask-split
// validation MSE where the physics term is supposed to act.
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

        FoldRecord record;
        record.candidate = config.name;
        record.physics_weight = config.loss.physics_weight;
        record.fold = fold_index;
        record.validation_mse = result.validation_mse;
        record.epochs = config.training.epochs;
        record.train_mse =
            evaluate_masked_mse(*model, dataset, fold.training, normalization)
                .full_mse;

        const MaskedMse validation_split = evaluate_masked_mse(
            *model, dataset, fold.validation, normalization);
        record.val_mse_small_angle = validation_split.small_angle_mse;
        record.small_angle_count = validation_split.small_angle_count;
        record.val_mse_large_angle = validation_split.large_angle_mse;
        record.large_angle_count = validation_split.large_angle_count;

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
        << "candidate,physics_weight,fold,train_mse,val_mse,baseline_mse,"
           "val_mse_small_angle,small_angle_count,val_mse_large_angle,"
           "large_angle_count,epochs\n";
    for (const FoldRecord& record : records) {
        csv << record.candidate << ',' << record.physics_weight << ','
            << record.fold << ',' << record.train_mse << ','
            << record.validation_mse << ',' << record.baseline_mse << ','
            << record.val_mse_small_angle << ',' << record.small_angle_count
            << ',' << record.val_mse_large_angle << ','
            << record.large_angle_count << ',' << record.epochs << '\n';
    }
}

std::string format_weight(float value) {
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

    TrialConfig base = make_config("phys", 0.25f, epochs);
    // Per-epoch gradient/update/activation statistics for stability analysis
    // (requested by the group); disable with --no-diagnostics.
    base.training.diagnostics.enabled = diagnostics;
    base.training.diagnostics.results_root = "results";
    base.training.diagnostics.experiment_name = "physics_weight_tuning";
    base.training.diagnostics.run_name = "sweep";
    base.training.diagnostics.training_dataset_path = kDatasetPath;
    base.training.diagnostics.validation_dataset_path = kDatasetPath;
    ParameterGrid grid(base);
    std::vector<NamedChoice<float>> choices;
    choices.reserve(values.size());
    for (float value : values) {
        choices.push_back(NamedChoice<float>{format_weight(value), value});
    }
    grid.add_choice<float>("physics-weight", choices,
                           [](TrialConfig& trial, const float& value) {
                               trial.loss.physics_weight = value;
                           });

    auto splitter = std::make_shared<RandomKFold>(kFoldCount, true, kSeed);
    auto runner = std::make_shared<RecordingRunner>(MPI_COMM_WORLD, records);
    CrossValidator validator(dataset, splitter, runner, MPI_COMM_WORLD, true);

    const SearchResult result = validator.tune(grid);

    if (rank == 0) {
        write_csv(csv_path, *records);
        std::cout << "\n============ PHYSICS-WEIGHT SWEEP RESULTS ============\n";
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
            // Log-ish spacing around the production default 0.25, with 0 as
            // the pure data-driven reference and 2 as a deliberately strong
            // prior that should visibly distort the fit if lambda matters.
            status = run_sweep(dataset,
                               {0.0f, 0.05f, 0.1f, 0.25f, 0.5f, 1.0f, 2.0f},
                               epochs,
                               "results/cross_validation/physics_weight_tuning/"
                               "sweep_physics.csv",
                               rank, diagnostics);
        } else if (rank == 0) {
            std::cout
                << "Usage: physics_weight_tuning <mode> [epochs] "
                   "[--no-diagnostics]\n"
                << "  sweep [epochs]   7-value lambda sweep, 5-fold CV "
                   "(default epochs: 100)\n"
                << "  Per-epoch gradient/weight diagnostics are written to\n"
                << "  results/physics_weight_tuning/sweep/ by default; disable\n"
                << "  with --no-diagnostics.\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "Physics-weight tuning failed on rank " << rank << ": "
                  << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return status;
}
