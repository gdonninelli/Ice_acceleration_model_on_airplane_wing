// Weight-regularization tuning experiment.
//
// Scientific question: does penalizing the weights improve generalization on
// this problem, and at which lambda? Two one-dimensional sweeps are run, L2
// alone and L1 alone, rather than a 2D grid: the measured train/validation gap
// is +0.000413 against a 0.001397 spread across folds, so there is no
// overfitting for the penalty to remove and an interaction term is not worth
// its cost.
//
// Selection metric: sample-weighted physical-unit validation MSE, but the
// conclusion is drawn from the PAIRED difference against lambda = 0, because
// the fold-to-fold spread is larger than any effect expected here. All
// candidates share one fold plan and one seed, which is what makes the pairing
// valid.
//
// Everything except lambda is held fixed: architecture, learning rate,
// physics weight, batch size, folds, epochs, seed.
//
// See CNN/experiments/regularization_tuning/README.md and cross_validation.md.

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
                        float l1_weight,
                        float l2_weight,
                        size_t epochs) {
    return TrialConfig{name,
                       make_blueprint(),
                       Recipes::adam(kLearningRate),
                       LossConfig{kPhysicsWeight, l1_weight, l2_weight},
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

bool is_weight_tensor(const LayerParameter& parameter) {
    return parameter.tensor && parameter.name == "weights";
}

// Physical-unit data MSE over an arbitrary index set, evaluated in chunks so a
// 1370-sample fold does not materialize as one 123 MB tensor. Run identically
// on every rank, so no reduction is needed.
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

// MSE of the predictor that always answers with the training-fold mean,
// scored on the validation fold, in the same physical units as
// Loss::physical_mse. make_batch standardizes by the training mean, so a
// prediction of zero in standardized space IS the training mean, and
// physical_mse against zeros is exactly mean((y - mean_train)^2).
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

// Mirrors CNNTrialRunner, and additionally records everything the paired
// analysis needs: the mean-predictor baseline, the physical-unit training MSE,
// the penalty actually reached, and the weight-norm evidence that lambda did
// something at all.
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
        const uint64_t seed = fold_seed(config.training.seed, fold_index);
        const NormalizationStats normalization =
            dataset.fit_normalization(fold.training);

        auto model = _model_factory.build(config, dataset.sdf_height(),
                                          dataset.sdf_width(),
                                          dataset.scalar_features(), seed,
                                          _communicator);
        const auto initial = snapshot_parameters(*model);

        const TrainingResult result = _trainer.fit(
            *model, dataset, fold.training, dataset, fold.validation,
            normalization, config.loss, config.training, seed, false);

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

std::string format_lambda(float value) {
    std::ostringstream text;
    text << std::scientific << std::setprecision(0) << value;
    return value == 0.0f ? std::string("0") : text.str();
}

// Convergence probe: one fold, no penalty, long run. Used only to size the
// epoch budget; it touches no test data.
int run_convergence(const Dataset& dataset, size_t epochs, int rank) {
    RandomKFold splitter(kFoldCount, true, kSeed);
    const auto folds = splitter.split(dataset.num_samples());
    const FoldIndices& fold = folds.front();
    const uint64_t seed = fold_seed(kSeed, 0);
    const NormalizationStats normalization =
        dataset.fit_normalization(fold.training);

    const TrialConfig config = make_config("convergence", 0.0f, 0.0f, epochs);
    ModelFactory factory;
    auto model = factory.build(config, dataset.sdf_height(), dataset.sdf_width(),
                               dataset.scalar_features(), seed, MPI_COMM_WORLD);
    Trainer trainer(MPI_COMM_WORLD);
    const TrainingResult result =
        trainer.fit(*model, dataset, fold.training, dataset, fold.validation,
                    normalization, config.loss, config.training, seed, false);

    if (rank == 0) {
        std::cout << std::setprecision(8)
                  << "epoch,train_objective,val_mse\n";
        for (const EpochMetrics& point : result.history) {
            std::cout << point.epoch << ',' << point.training_objective << ','
                      << point.validation_mse << '\n';
        }
        std::cout << "# final validation MSE: " << result.validation_mse
                  << std::endl;
    }
    return 0;
}

int run_sweep(const Dataset& dataset,
              const std::string& axis,
              const std::vector<float>& values,
              size_t epochs,
              const std::string& csv_path,
              int rank) {
    const bool l2_axis = axis == "l2";
    auto records = std::make_shared<std::vector<FoldRecord>>();

    TrialConfig base = make_config("reg", 0.0f, 0.0f, epochs);
    ParameterGrid grid(base);
    std::vector<NamedChoice<float>> choices;
    choices.reserve(values.size());
    for (float value : values) {
        choices.push_back(NamedChoice<float>{format_lambda(value), value});
    }
    if (l2_axis) {
        grid.add_choice<float>("l2", choices,
                               [](TrialConfig& trial, const float& value) {
                                   trial.loss.l2_weight = value;
                               });
    } else {
        grid.add_choice<float>("l1", choices,
                               [](TrialConfig& trial, const float& value) {
                                   trial.loss.l1_weight = value;
                               });
    }

    auto splitter = std::make_shared<RandomKFold>(kFoldCount, true, kSeed);
    auto runner = std::make_shared<RecordingRunner>(MPI_COMM_WORLD, records);
    CrossValidator validator(dataset, splitter, runner, MPI_COMM_WORLD, true);

    const SearchResult result = validator.tune(grid);

    if (rank == 0) {
        write_csv(csv_path, *records);
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
        const size_t epochs = argc > 2 ? std::stoul(argv[2]) : 100;
        Dataset dataset("dataset/cnn_dataset_train.npz");

        if (mode == "converge") {
            status = run_convergence(dataset, epochs, rank);
        } else if (mode == "sweep-l2") {
            // Log-spaced over the band where the measured balancing lambda2
            // sits (0.0762 at init, 7e-4 to 1e-3 later), plus 0 as reference.
            status = run_sweep(dataset, "l2",
                               {0.0f, 1e-4f, 3.16e-4f, 1e-3f, 3.16e-3f, 1e-2f,
                                3.16e-2f, 1e-1f},
                               epochs,
                               "results/cross_validation/regularization_tuning/"
                               "sweep_l2.csv",
                               rank);
        } else if (mode == "sweep-l1") {
            // Centred on the measured median |grad_data| rather than reused
            // from L2: the L1 gradient is lambda1*sign(w), independent of |w|,
            // so lambda1* IS the median |grad_data| (2.13e-5 at epoch 100)
            // rather than grad/(2|w|). Same 3-decade width as the L2 sweep, so
            // the extremes sit at 1/30 and 30x the data gradient.
            status = run_sweep(dataset, "l1",
                               {0.0f, 6.75e-7f, 2.13e-6f, 6.75e-6f, 2.13e-5f,
                                6.75e-5f, 2.13e-4f, 6.75e-4f},
                               epochs,
                               "results/cross_validation/regularization_tuning/"
                               "sweep_l1.csv",
                               rank);
        } else if (rank == 0) {
            std::cout << "Usage: regularization_tuning <mode> <epochs>\n"
                      << "  converge <epochs>   one fold, lambda = 0, prints the "
                         "train/val curve\n"
                      << "  sweep-l2 <epochs>   L2 sweep, lambda1 = 0\n"
                      << "  sweep-l1 <epochs>   L1 sweep, lambda2 = 0\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "Regularization tuning failed on rank " << rank << ": "
                  << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return status;
}
