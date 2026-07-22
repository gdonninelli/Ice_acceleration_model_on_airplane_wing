#include "core/Loss.hpp"
#include "data/Dataset.hpp"
#include "layers/ConcatenateLayer.hpp"
#include "layers/DenseLayer.hpp"
#include "model/ModelFactory.hpp"
#include "tuning/CrossValidator.hpp"
#include "tuning/SearchSpace.hpp"
#include "tuning/TrialConfig.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <numbers>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
static_assert(Trainer::kHistoryIntervalEpochs == 10);

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual,
                   double expected,
                   double tolerance,
                   const std::string& message) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + ": expected " +
                                 std::to_string(expected) + ", got " +
                                 std::to_string(actual));
    }
}

template <typename Function>
void require_throws(Function function, const std::string& message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

TrialConfig minimal_trial(float physics_weight = 0.25f) {
    ModelBlueprint blueprint;
    blueprint.feature_layers = {Recipes::flatten()};
    blueprint.head_layers = {Recipes::dense(1)};
    return TrialConfig{"test", blueprint, Recipes::adam(1e-3f),
                       LossConfig{physics_weight},
                       TrainingConfig{1, 2, 1.0f, 7, false}, {}};
}

Dataset tiny_dataset(size_t samples) {
    std::vector<float> sdf(samples);
    std::vector<float> scalars(samples * 2);
    std::vector<float> targets(samples);
    for (size_t index = 0; index < samples; ++index) {
        sdf[index] = static_cast<float>(index);
        scalars[index * 2] = 100000.0f + static_cast<float>(index);
        scalars[index * 2 + 1] = static_cast<float>(index) - 2.0f;
        targets[index] = static_cast<float>(index) * 0.1f;
    }
    return Dataset(std::move(sdf), std::move(scalars), std::move(targets), 1, 1);
}

void test_random_kfold() {
    RandomKFold splitter(3, true, 91);
    const auto first = splitter.split(10);
    const auto second = splitter.split(10);
    require(first.size() == 3 && second.size() == 3,
            "RandomKFold returned the wrong fold count");

    std::set<size_t> validation_samples;
    for (size_t fold = 0; fold < first.size(); ++fold) {
        require(first[fold].validation == second[fold].validation,
                "RandomKFold is not deterministic");
        std::set<size_t> training(first[fold].training.begin(),
                                  first[fold].training.end());
        for (size_t index : first[fold].validation) {
            require(!training.contains(index), "Fold training/validation overlap");
            require(validation_samples.insert(index).second,
                    "Sample appears in multiple validation folds");
        }
    }
    require(validation_samples.size() == 10,
            "Validation folds do not cover every sample");
}

void test_fold_normalization_and_schema() {
    Dataset dataset({0.0f, 2.0f, 100.0f},
                    {10.0f, -10.0f, 30.0f, 10.0f, 1000.0f, 90.0f},
                    {1.0f, 3.0f, 100.0f}, 1, 1);
    const std::vector<size_t> training{0, 1};
    const NormalizationStats stats = dataset.fit_normalization(training);
    require_close(stats.sdf_mean, 1.0, 1e-6,
                  "SDF normalization leaked validation data");
    require_close(stats.target_mean, 2.0, 1e-6,
                  "Target normalization leaked validation data");

    const std::vector<size_t> validation{2};
    const DataBatch batch = dataset.make_batch(validation, stats);
    require_close(batch.alpha_radians->get_data()[0], std::numbers::pi / 2.0,
                  1e-6, "AoA column or degree conversion is incorrect");
    require(batch.scalars->get_data()[0] > 10.0f,
            "Validation Reynolds value was not transformed with training statistics");

    require_throws(
        [] {
            Dataset invalid({std::numeric_limits<float>::quiet_NaN()},
                            {1.0f, 2.0f}, {1.0f}, 1, 1);
        },
        "Dataset accepted a non-finite value");
}

void test_physical_simm_scaling() {
    auto predictions = std::make_shared<Tensor>(std::vector<size_t>{1, 1});
    auto targets = std::make_shared<Tensor>(std::vector<size_t>{1, 1});
    auto alphas = std::make_shared<Tensor>(std::vector<size_t>{1, 1});
    predictions->get_data()[0] = 0.0f;
    targets->get_data()[0] = 0.0f;
    alphas->get_data()[0] = 0.0f;

    require_close(Loss::simm_forward(predictions, targets, alphas, 1.0f,
                                     1.0f, 2.0f),
                  0.25, 1e-6, "SIMM physics target is not standardized correctly");
    const auto gradient = Loss::simm_backward(predictions, targets, alphas,
                                              1.0f, 1.0f, 2.0f);
    require_close(gradient->get_data()[0], 1.0, 1e-6,
                  "SIMM physical scaling gradient is incorrect");
}

void test_parameter_grid_and_fresh_models() {
    ParameterGrid grid(minimal_trial());
    grid.add_choice<float>(
        "physics", {{"low", 0.1f}, {"high", 0.5f}},
        [](TrialConfig& config, float value) {
            config.loss.physics_weight = value;
        });
    grid.add_choice<OptimizerRecipe>(
        "optimizer",
        {{"adam-slow", Recipes::adam(1e-4f)},
         {"adam-fast", Recipes::adam(1e-3f)}},
        [](TrialConfig& config, const OptimizerRecipe& value) {
            config.optimizer = value;
        });
    require(grid.candidates().size() == 4,
            "ParameterGrid did not create the Cartesian product");

    ParameterGrid retryable(minimal_trial());
    require_throws(
        [&] {
            retryable.add_choice<float>(
                "retry", {{"", 0.1f}},
                [](TrialConfig&, float) {});
        },
        "ParameterGrid accepted an empty choice label");
    retryable.add_choice<float>(
        "retry", {{"valid", 0.1f}}, [](TrialConfig&, float) {});
    require(retryable.candidates().size() == 1,
            "Failed ParameterGrid addition corrupted the axis registry");

    ModelFactory factory;
    TrialConfig config = minimal_trial();
    auto first = factory.build(config, 2, 2, 2, 12, MPI_COMM_WORLD);
    auto second = factory.build(config, 2, 2, 2, 12, MPI_COMM_WORLD);
    require(!first->parameters().empty(), "Factory model has no parameters");
    float* first_weight = first->parameters()[0].tensor->get_data();
    float* second_weight = second->parameters()[0].tensor->get_data();
    require_close(first_weight[0], second_weight[0], 1e-7,
                  "Equal seeds did not reproduce initialization");
    first_weight[0] += 1.0f;
    require(std::abs(first_weight[0] - second_weight[0]) > 0.5f,
            "Factory reused model parameter state");

    auto high_seed = factory.build(config, 2, 2, 2,
                                   (uint64_t{1} << 40U) + 12, MPI_COMM_WORLD);
    require(std::abs(second_weight[0] -
                     high_seed->parameters()[0].tensor->get_data()[0]) > 1e-7f,
            "High seed bits did not affect initialization");

    require_throws(
        [] { Recipes::flatten().infer_shape({4}); },
        "Flatten shape inference accepted a non-spatial input");

    DenseLayer dense(2, 1, 3);
    auto dense_input = std::make_shared<Tensor>(std::vector<size_t>{1, 2});
    dense.forward({dense_input});
    auto wrong_gradient = std::make_shared<Tensor>(std::vector<size_t>{1, 2});
    require_throws(
        [&] { dense.backward(wrong_gradient); },
        "Dense backward accepted an incompatible gradient shape");
    auto wrong_input = std::make_shared<Tensor>(std::vector<size_t>{1, 3});
    require_throws([&] { dense.forward({wrong_input}); },
                   "Dense forward accepted an incompatible input");
    auto formerly_valid_gradient =
        std::make_shared<Tensor>(std::vector<size_t>{1, 1});
    require_throws(
        [&] { dense.backward(formerly_valid_gradient); },
        "Dense backward reused stale caches after a failed forward");

    ConcatenateLayer concatenate(3);
    auto features = std::make_shared<Tensor>(std::vector<size_t>{1, 1});
    auto wrong_scalars = std::make_shared<Tensor>(std::vector<size_t>{1, 2});
    require_throws(
        [&] { concatenate.forward({features, wrong_scalars}); },
        "ConcatenateLayer ignored its configured scalar width");

    TrialConfig scalar_mismatch = minimal_trial();
    scalar_mismatch.model.scalar_features = 3;
    require_throws(
        [&] {
            factory.build(scalar_mismatch, 2, 2, 2, 1, MPI_COMM_WORLD);
        },
        "ModelFactory accepted a scalar width that differs from the dataset");
}

void test_trainer_with_partial_batches() {
    Dataset dataset = tiny_dataset(7);
    const std::vector<size_t> training{0, 1, 2, 3, 4};
    const std::vector<size_t> validation{5, 6};
    const NormalizationStats normalization =
        dataset.fit_normalization(training);
    TrialConfig config = minimal_trial(0.0f);
    config.training = TrainingConfig{10, 3, 1.0f, 19, true};

    ModelFactory factory;
    auto model = factory.build(config, 1, 1, 2, 19, MPI_COMM_WORLD);
    Trainer trainer(MPI_COMM_WORLD);
    const TrainingResult result = trainer.fit(
        *model, dataset, training, dataset, validation, normalization,
        config.loss, config.training, 19, false);
    require(std::isfinite(result.training_objective),
            "Trainer returned a non-finite objective");
    require(std::isfinite(result.validation_mse),
            "Trainer returned a non-finite validation MSE");
    require(result.training_samples == training.size() &&
                result.validation_samples == validation.size(),
            "Trainer reported incorrect sample counts");
    require(result.history.size() == 1 && result.history[0].epoch == 10,
            "Trainer did not record the fixed 10-epoch history checkpoint");
    require_close(result.history[0].training_objective,
                  result.training_objective, 1e-7,
                  "Final training objective differs from epoch history");
    require_close(result.history[0].validation_mse,
                  result.validation_mse, 1e-7,
                  "Final validation MSE differs from epoch history");

    auto reference_model = factory.build(config, 1, 1, 2, 19, MPI_COMM_SELF);
    Trainer reference_trainer(MPI_COMM_SELF);
    const TrainingResult reference = reference_trainer.fit(
        *reference_model, dataset, training, dataset, validation, normalization,
        config.loss, config.training, 19, false);
    require_close(result.training_objective, reference.training_objective, 1e-5,
                  "Distributed training objective differs from serial training");
    require_close(result.validation_mse, reference.validation_mse, 1e-5,
                  "Distributed validation differs from serial validation");
    const auto distributed_parameters = model->parameters();
    const auto reference_parameters = reference_model->parameters();
    require(distributed_parameters.size() == reference_parameters.size(),
            "Distributed and serial models expose different parameters");
    for (size_t parameter = 0; parameter < distributed_parameters.size(); ++parameter) {
        require(distributed_parameters[parameter].tensor->size() ==
                    reference_parameters[parameter].tensor->size(),
                "Distributed and serial parameter shapes differ");
        for (size_t value = 0;
             value < distributed_parameters[parameter].tensor->size(); ++value) {
            require_close(
                distributed_parameters[parameter].tensor->get_data()[value],
                reference_parameters[parameter].tensor->get_data()[value], 1e-5,
                "MPI weighted gradient update differs from serial update");
        }
    }
}

void test_generic_checkpoint_round_trip() {
    TrialConfig config = minimal_trial();
    ModelFactory factory;
    auto source = factory.build(config, 2, 2, 2, 4, MPI_COMM_WORLD);
    const auto source_parameters = source->parameters();
    for (size_t parameter = 0; parameter < source_parameters.size(); ++parameter) {
        for (size_t value = 0; value < source_parameters[parameter].tensor->size();
             ++value) {
            source_parameters[parameter].tensor->get_data()[value] =
                static_cast<float>(parameter * 100 + value);
        }
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("cnn_cv_weights_test_" + std::to_string(rank) + ".bin");
    source->export_weights(path.string());
    auto restored = factory.build(config, 2, 2, 2, 99, MPI_COMM_WORLD);
    restored->import_weights(path.string());
    std::filesystem::remove(path);

    const auto restored_parameters = restored->parameters();
    require(source_parameters.size() == restored_parameters.size(),
            "Checkpoint changed the parameter count");
    for (size_t parameter = 0; parameter < source_parameters.size(); ++parameter) {
        for (size_t value = 0; value < source_parameters[parameter].tensor->size();
             ++value) {
            require_close(source_parameters[parameter].tensor->get_data()[value],
                          restored_parameters[parameter].tensor->get_data()[value],
                          0.0, "Generic checkpoint round trip changed a value");
        }
    }
}

class FakeTrialRunner final : public TrialRunner {
public:
    FoldMetrics run(const TrialConfig& config,
                    const Dataset&,
                    const FoldIndices& fold,
                    size_t fold_index) const override {
        std::vector<EpochMetrics> history;
        for (size_t epoch = Trainer::kHistoryIntervalEpochs;
             epoch <= config.training.epochs;
             epoch += Trainer::kHistoryIntervalEpochs) {
            history.push_back(EpochMetrics{
                epoch, 1.0, config.loss.physics_weight});
        }
        return FoldMetrics{fold_index,
                           1.0,
                           config.loss.physics_weight,
                           fold.training.size(),
                           fold.validation.size(),
                           std::move(history)};
    }
};

class InvalidTrialRunner final : public TrialRunner {
public:
    FoldMetrics run(const TrialConfig&,
                    const Dataset&,
                    const FoldIndices& fold,
                    size_t fold_index) const override {
        return FoldMetrics{fold_index,
                           1.0,
                           std::numeric_limits<double>::quiet_NaN(),
                           fold.training.size(),
                           fold.validation.size(),
                           {}};
    }
};

class OverlappingSplitter final : public FoldSplitter {
public:
    std::vector<FoldIndices> split(size_t) const override {
        return {{{0, 1}, {1}}, {{1}, {0}}};
    }
};

class CountingSplitter final : public FoldSplitter {
public:
    std::vector<FoldIndices> split(size_t sample_count) const override {
        ++_calls;
        return RandomKFold(3, true, 17).split(sample_count);
    }

    size_t calls() const { return _calls; }

private:
    mutable size_t _calls = 0;
};

class DivergentTrialRunner final : public TrialRunner {
public:
    FoldMetrics run(const TrialConfig&,
                    const Dataset&,
                    const FoldIndices& fold,
                    size_t fold_index) const override {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return FoldMetrics{fold_index,
                           1.0,
                           0.2 + static_cast<double>(rank),
                           fold.training.size(),
                           fold.validation.size(),
                           {}};
    }
};

void test_cross_validator_selection() {
    Dataset dataset = tiny_dataset(6);
    ParameterGrid grid(minimal_trial());
    grid.add_choice<float>(
        "score", {{"worse", 0.7f}, {"better", 0.2f}},
        [](TrialConfig& config, float value) {
            config.loss.physics_weight = value;
        });
    CrossValidator validator(dataset, std::make_shared<RandomKFold>(3, true, 2),
                             std::make_shared<FakeTrialRunner>(), MPI_COMM_WORLD,
                             false);
    const SearchResult result = validator.tune(grid);
    require(result.candidates.size() == 2,
            "CrossValidator returned the wrong candidate count");
    require_close(result.best().mean_validation_mse, 0.2, 1e-6,
                  "CrossValidator selected the wrong candidate");

    TrialConfig history_config = minimal_trial(0.2f);
    history_config.training.epochs = 20;
    const CandidateResult history_result = validator.evaluate(history_config);
    require(history_result.folds.size() == 3,
            "CrossValidator returned the wrong number of history folds");
    for (const auto& fold : history_result.folds) {
        require(fold.history.size() == 2 &&
                    fold.history[0].epoch == 10 &&
                    fold.history[1].epoch == 20,
                "CrossValidator did not retain the per-fold epoch history");
    }

    auto counting_splitter = std::make_shared<CountingSplitter>();
    CrossValidator fixed_folds(dataset, counting_splitter,
                               std::make_shared<FakeTrialRunner>(),
                               MPI_COMM_WORLD, false);
    fixed_folds.tune(grid);
    require(counting_splitter->calls() == 1,
            "CrossValidator regenerated folds for each candidate");

    CrossValidator invalid_metrics(
        dataset, std::make_shared<RandomKFold>(3, true, 2),
        std::make_shared<InvalidTrialRunner>(), MPI_COMM_WORLD, false);
    const SearchResult failed = invalid_metrics.tune(grid);
    require(failed.candidates.size() == 2 && !failed.candidates[0].success,
            "CrossValidator did not retain failed candidates");
    require_throws([&] { failed.best(); },
                   "SearchResult returned a best candidate when all failed");

    CrossValidator invalid_folds(
        dataset, std::make_shared<OverlappingSplitter>(),
        std::make_shared<FakeTrialRunner>(), MPI_COMM_WORLD, false);
    require_throws([&] { invalid_folds.evaluate(minimal_trial()); },
                   "CrossValidator accepted overlapping fold indices");

    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size > 1) {
        CrossValidator divergent(
            dataset, std::make_shared<RandomKFold>(3, true, 2),
            std::make_shared<DivergentTrialRunner>(), MPI_COMM_WORLD, false);
        const SearchResult divergent_result = divergent.tune(grid);
        require(!divergent_result.candidates[0].success,
                "CrossValidator accepted rank-divergent metrics");
    }
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    try {
        test_random_kfold();
        test_fold_normalization_and_schema();
        test_physical_simm_scaling();
        test_parameter_grid_and_fresh_models();
        test_trainer_with_partial_batches();
        test_generic_checkpoint_round_trip();
        test_cross_validator_selection();
        std::cout << "All cross-validation tests passed." << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << std::endl;
        MPI_Finalize();
        return 1;
    }
    MPI_Finalize();
    return 0;
}
