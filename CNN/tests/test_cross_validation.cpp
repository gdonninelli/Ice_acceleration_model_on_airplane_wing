#include "core/Loss.hpp"
#include "data/Dataset.hpp"
#include "layers/ConcatenateLayer.hpp"
#include "layers/DenseLayer.hpp"
#include "layers/DropoutLayer.hpp"
#include "layers/ReLULayer.hpp"
#include "model/ModelFactory.hpp"
#include "optimizers/AdamOptimizer.hpp"
#include "optimizers/LRScheduler.hpp"
#include "training/TrainingDiagnostics.hpp"
#include "tuning/CrossValidator.hpp"
#include "tuning/SearchSpace.hpp"
#include "tuning/TrialConfig.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

void test_diagnostics_math_and_activation_capture() {
    RunningStatistics first;
    first.add(1.0);
    first.add(2.0);
    RunningStatistics second;
    second.add(3.0);
    first.merge(second);
    require(first.count() == 3, "Streaming statistics count is incorrect");
    require_close(first.mean(), 2.0, 1e-12,
                  "Streaming statistics mean is incorrect");
    require_close(first.population_variance(), 2.0 / 3.0, 1e-12,
                  "Streaming population variance is incorrect");
    require_close(first.minimum(), 1.0, 0.0,
                  "Streaming minimum is incorrect");
    require_close(first.maximum(), 3.0, 0.0,
                  "Streaming maximum is incorrect");

    const float norm_values[]{3.0f, 4.0f};
    require_close(l2_norm(norm_values, 2), 5.0, 0.0,
                  "Gradient L2 norm calculation is incorrect");
    const float before[]{3.0f, 4.0f};
    const float after[]{3.0f, 0.0f};
    const UpdateMeasurement update = measure_update(before, after, 2);
    require_close(update.pre_update_norm, 5.0, 0.0,
                  "Pre-update norm is incorrect");
    require_close(update.update_norm, 4.0, 0.0,
                  "Actual update delta is incorrect");
    require_close(update.ratio, 0.8, 1e-12,
                  "Actual update ratio is incorrect");
    const float zeros[]{0.0f, 0.0f};
    const float tiny_update[]{1e-13f, 0.0f};
    const UpdateMeasurement near_zero =
        measure_update(zeros, tiny_update, 2);
    require(near_zero.denominator_near_zero,
            "Zero update-ratio denominator was not flagged");
    require_close(near_zero.ratio, 0.1, 1e-6,
                  "Update-ratio epsilon handling is incorrect");

    CNNModel model(std::make_unique<AdamOptimizer>(), MPI_COMM_SELF);
    model.add_conv_layer(std::make_unique<ReLULayer>());
    size_t captures = 0;
    model.set_activation_observer(
        [&](size_t layer_index, const std::string& layer_name,
            const Tensor& pre, const Tensor& post) {
            require(layer_index == 0 && layer_name == "ReLULayer",
                    "Activation observer received unstable layer metadata");
            require_close(pre.get_data()[0], -1.0, 0.0,
                          "Pre-activation capture is incorrect");
            require_close(post.get_data()[0], 0.0, 0.0,
                          "Post-activation capture is incorrect");
            ++captures;
        });
    auto input = std::make_shared<Tensor>(std::vector<size_t>{1, 2});
    input->get_data()[0] = -1.0f;
    input->get_data()[1] = 2.0f;
    model.forward(input, nullptr);
    require(captures == 1, "Activation observer did not capture the training pass");
    const auto layers = model.layer_info();
    require(layers.size() == 1 && layers[0].index == 0 && layers[0].activation,
            "Model layer ordering metadata is not deterministic");
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not read test artifact: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

void test_training_diagnostics_artifacts() {
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const auto root = std::filesystem::temp_directory_path() /
        ("cnn_diagnostics_test_" + std::to_string(world_size));
    if (rank == 0) {
        std::filesystem::remove_all(root);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    Dataset dataset = tiny_dataset(7);
    const std::vector<size_t> training{0, 1, 2, 3, 4};
    const std::vector<size_t> validation{5, 6};
    const NormalizationStats normalization =
        dataset.fit_normalization(training);
    ModelBlueprint blueprint;
    blueprint.feature_layers = {Recipes::activation("relu"), Recipes::flatten()};
    blueprint.head_layers = {Recipes::dense(1)};
    TrialConfig config{"diagnostics candidate", blueprint, Recipes::adam(1e-3f),
                       LossConfig{0.0f},
                       TrainingConfig{3, 3, 1.0f, 31, false}, {}};
    config.training.validation_interval = 2;
    config.training.diagnostics.enabled = true;
    config.training.diagnostics.results_root = root.string();
    config.training.diagnostics.experiment_name = "experiment";
    config.training.diagnostics.run_name = "run";
    config.training.diagnostics.histogram_bins = 8;

    ModelFactory factory;
    auto model = factory.build(config, 1, 1, 2, 31, MPI_COMM_WORLD);
    Trainer trainer(MPI_COMM_WORLD);
    TrainingRunContext context;
    context.random_seed = 31;
    const TrainingResult diagnostic_result = trainer.fit(
        *model, dataset, training, dataset, validation, normalization,
        config.loss, config.training, 31, false, context, &config);
    MPI_Barrier(MPI_COMM_WORLD);

    std::string artifact_error;
    if (rank == 0) {
        try {
            const auto directory = root / "experiment" / "run";
            require(std::filesystem::exists(directory / "metadata.json"),
                    "Diagnostics metadata was not written");
            const std::string metadata =
                read_text_file(directory / "metadata.json");
            require(metadata.find("\"early_stopping_min_epochs\": 20") !=
                        std::string::npos &&
                        metadata.find("\"early_stopping_patience\": 20") !=
                        std::string::npos,
                    "Early-stopping policy is missing from diagnostics metadata");
            require(metadata.find("\"batch_construction\": \"balanced\"") !=
                        std::string::npos &&
                        metadata.find("\"early_stopping_policy\": \"patience\"") !=
                        std::string::npos,
                    "Training strategy names are missing from diagnostics metadata");
            const std::string epoch_metrics =
                read_text_file(directory / "epoch_metrics.csv");
            require(epoch_metrics.starts_with(
                        "epoch,train_objective,training_physical_mse,"
                        "validation_physical_mse,samples,batches,"
                        "configured_learning_rate,effective_learning_rate\n"),
                    "Epoch metrics schema is not explicit about physical MSE");
            require(std::count(epoch_metrics.begin(), epoch_metrics.end(), '\n') == 4,
                    "Epoch metrics did not record every training epoch");
            require(epoch_metrics.find(",,") == std::string::npos,
                    "Epoch metrics omitted a per-epoch physical MSE");
            const std::string activations =
                read_text_file(directory / "activation_statistics.csv");
            require(activations.find(",pre_activation,5,") != std::string::npos &&
                        activations.find(",post_activation,5,") != std::string::npos,
                    "Activation statistics were not globally aggregated");
            const std::string gradients =
                read_text_file(directory / "gradient_norms.csv");
            require(gradients.find(",\"all\",") != std::string::npos &&
                        gradients.find(",2\n") != std::string::npos,
                    "Epoch gradient aggregation or optimizer-step count is missing");
            const std::string updates =
                read_text_file(directory / "parameter_update_ratios.csv");
            require(updates.find("near_zero_denominator_steps") != std::string::npos,
                    "Parameter update-ratio output is incomplete");
            const std::string histograms =
                read_text_file(directory / "activation_histograms.csv");
            require(histograms.find(",5\n") != std::string::npos,
                    "Activation histogram population is incorrect");

            TrainingRunContext first_fold;
            first_fold.mode = "cross_validation";
            first_fold.candidate_index = 0;
            first_fold.fold_index = 0;
            TrainingRunContext second_fold = first_fold;
            second_fold.fold_index = 1;
            require(diagnostics_run_directory(config.training.diagnostics, first_fold) !=
                        diagnostics_run_directory(config.training.diagnostics, second_fold),
                    "Candidate/fold diagnostics paths collide");
        } catch (const std::exception& error) {
            artifact_error = error.what();
        }
    }
    int artifact_failed = artifact_error.empty() ? 0 : 1;
    MPI_Bcast(&artifact_failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    require(artifact_failed == 0,
            artifact_error.empty() ? "Diagnostics artifact verification failed on rank zero"
                                   : artifact_error);

    CrossValidator diagnostic_cv(
        dataset, std::make_shared<RandomKFold>(2, true, 31),
        std::make_shared<CNNTrialRunner>(MPI_COMM_WORLD), MPI_COMM_WORLD, false);
    const CandidateResult cv_result = diagnostic_cv.evaluate(config);
    require(cv_result.folds.size() == 2,
            "Diagnostics cross-validation returned the wrong fold count");
    MPI_Barrier(MPI_COMM_WORLD);
    int cv_artifacts_missing = 0;
    if (rank == 0) {
        const auto session = root / "experiment" / "run";
        cv_artifacts_missing =
            (!std::filesystem::exists(session / "candidate_000" / "fold_000" /
                                      "epoch_metrics.csv") ||
             !std::filesystem::exists(session / "candidate_000" / "fold_001" /
                                      "epoch_metrics.csv") ||
             !std::filesystem::exists(session / "cv_summary.csv"))
                ? 1 : 0;
    }
    MPI_Bcast(&cv_artifacts_missing, 1, MPI_INT, 0, MPI_COMM_WORLD);
    require(cv_artifacts_missing == 0,
            "Cross-validation candidate/fold artifacts are not separated");

    TrialConfig disabled = config;
    disabled.training.diagnostics.enabled = false;
    disabled.training.diagnostics.run_name = "disabled";
    auto disabled_model =
        factory.build(disabled, 1, 1, 2, 31, MPI_COMM_WORLD);
    const TrainingResult disabled_result = trainer.fit(
        *disabled_model, dataset, training, dataset, validation,
        normalization, disabled.loss, disabled.training, 31, false,
        context, &disabled);
    require_close(disabled_result.training_objective,
                  diagnostic_result.training_objective, 1e-7,
                  "Enabled diagnostics changed the training objective");
    require_close(disabled_result.validation_mse,
                  diagnostic_result.validation_mse, 1e-7,
                  "Enabled diagnostics changed validation behavior");
    const auto diagnostic_parameters = model->parameters();
    const auto disabled_parameters = disabled_model->parameters();
    require(diagnostic_parameters.size() == disabled_parameters.size(),
            "Enabled diagnostics changed parameter ordering");
    for (size_t parameter = 0; parameter < diagnostic_parameters.size(); ++parameter) {
        for (size_t value = 0;
             value < diagnostic_parameters[parameter].tensor->size(); ++value) {
            require_close(
                diagnostic_parameters[parameter].tensor->get_data()[value],
                disabled_parameters[parameter].tensor->get_data()[value], 1e-7,
                "Enabled diagnostics changed a model update");
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int disabled_artifact_exists = 0;
    if (rank == 0) {
        disabled_artifact_exists =
            std::filesystem::exists(root / "experiment" / "disabled") ? 1 : 0;
        if (std::getenv("CNN_KEEP_TEST_ARTIFACTS") == nullptr) {
            std::filesystem::remove_all(root);
        }
    }
    MPI_Bcast(&disabled_artifact_exists, 1, MPI_INT, 0, MPI_COMM_WORLD);
    require(disabled_artifact_exists == 0,
            "Disabled diagnostics created result artifacts");
    MPI_Barrier(MPI_COMM_WORLD);
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

void test_historical_range_tail_batching() {
    Dataset dataset = tiny_dataset(7);
    const std::vector<size_t> training{0, 1, 2, 3, 4, 5, 6};
    const std::vector<size_t> validation{5, 6};
    const NormalizationStats normalization =
        dataset.fit_normalization(training);
    TrialConfig balanced = minimal_trial(0.0f);
    balanced.training.epochs = 1;
    balanced.training.global_batch_size = 3;
    balanced.training.shuffle = false;

    TrialConfig range_tail = balanced;
    range_tail.training.batch_construction = BatchConstruction::RangeTail;

    ModelFactory factory;
    Trainer trainer(MPI_COMM_WORLD);
    auto balanced_model = factory.build(
        balanced, 1, 1, 2, balanced.training.seed, MPI_COMM_WORLD);
    auto range_tail_model = factory.build(
        range_tail, 1, 1, 2, range_tail.training.seed, MPI_COMM_WORLD);
    const TrainingResult balanced_result = trainer.fit(
        *balanced_model, dataset, training, dataset, validation, normalization,
        balanced.loss, balanced.training, balanced.training.seed, false);
    const TrainingResult range_tail_result = trainer.fit(
        *range_tail_model, dataset, training, dataset, validation, normalization,
        range_tail.loss, range_tail.training, range_tail.training.seed, false);

    require(std::abs(balanced_result.training_mse -
                     range_tail_result.training_mse) > 1e-8,
            "Historical range-tail batching matched balanced batch updates");
}

void test_early_stopping_policy() {
    Dataset dataset = tiny_dataset(7);
    const std::vector<size_t> training{0, 1, 2, 3, 4};
    const std::vector<size_t> validation{5, 6};
    const NormalizationStats normalization =
        dataset.fit_normalization(training);

    ModelFactory factory;
    Trainer trainer(MPI_COMM_WORLD);

    TrialConfig warmup = minimal_trial(0.0f);
    warmup.training.epochs = 3;
    warmup.training.early_stopping = true;
    warmup.training.early_stopping_min_epochs = 20;
    warmup.training.early_stopping_patience = 1;
    auto warmup_model = factory.build(warmup, 1, 1, 2,
                                      warmup.training.seed, MPI_COMM_WORLD);
    const TrainingResult result = trainer.fit(
        *warmup_model, dataset, training, dataset, validation, normalization,
        warmup.loss, warmup.training, warmup.training.seed, false);
    require(result.epochs_completed == warmup.training.epochs &&
                !result.stopped_early,
            "Early stopping ignored its minimum-epoch warm-up");

    TrialConfig improving = minimal_trial(0.0f);
    improving.training.epochs = 5;
    improving.training.early_stopping = true;
    improving.training.early_stopping_min_epochs = 0;
    improving.training.early_stopping_patience = 1;
    improving.training.max_overfit_ratio = 0.0;
    auto improving_model = factory.build(
        improving, 1, 1, 2, improving.training.seed, MPI_COMM_WORLD);
    const TrainingResult immediate = trainer.fit(
        *improving_model, dataset, training, dataset, validation, normalization,
        improving.loss, improving.training, improving.training.seed, false);
    require(immediate.epochs_completed == improving.training.epochs &&
                !immediate.stopped_early &&
                immediate.best_epoch == improving.training.epochs,
            "Validation improvements did not reset the overfit streak");

    Dataset overfit_dataset(
        {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        {100000.0f, -2.0f, 100001.0f, -1.0f, 100002.0f, 0.0f,
         100003.0f, 1.0f, 100004.0f, 2.0f, 100005.0f, 3.0f,
         100006.0f, 4.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 100.0f, 100.0f}, 1, 1);
    const NormalizationStats overfit_normalization =
        overfit_dataset.fit_normalization(training);
    TrialConfig overfit_config = minimal_trial(0.0f);
    overfit_config.training.epochs = 5;
    overfit_config.training.validation_interval = 1;
    overfit_config.training.early_stopping = true;
    overfit_config.training.early_stopping_min_epochs = 0;
    overfit_config.training.early_stopping_patience = 2;
    auto overfit_model = factory.build(
        overfit_config, 1, 1, 2, overfit_config.training.seed, MPI_COMM_WORLD);
    const TrainingResult overfit = trainer.fit(
        *overfit_model, overfit_dataset, training, overfit_dataset, validation,
        overfit_normalization, overfit_config.loss, overfit_config.training,
        overfit_config.training.seed, false);
    require(overfit.stopped_early && overfit.epochs_completed == 3 &&
                overfit.best_epoch == 1 && overfit.history.size() == 3,
            "Early stopping did not require consecutive patience epochs");
    require_close(overfit.validation_mse, overfit.history.front().validation_mse,
                  1e-5, "Early stopping did not restore the best checkpoint");

    TrialConfig historical = overfit_config;
    historical.training.early_stopping_policy =
        EarlyStoppingPolicy::FirstRatioExceeded;
    auto historical_model = factory.build(
        historical, 1, 1, 2, historical.training.seed, MPI_COMM_WORLD);
    const TrainingResult first_crossing = trainer.fit(
        *historical_model, overfit_dataset, training, overfit_dataset,
        validation, overfit_normalization, historical.loss,
        historical.training, historical.training.seed, false);
    require(first_crossing.stopped_early &&
                first_crossing.epochs_completed == 1,
            "Historical early stopping did not stop at the first ratio crossing");

    TrialConfig invalid = overfit_config;
    invalid.training.early_stopping_patience = 0;
    auto invalid_model = factory.build(invalid, 1, 1, 2,
                                       invalid.training.seed, MPI_COMM_WORLD);
    require_throws(
        [&] {
            trainer.fit(*invalid_model, dataset, training, dataset, validation,
                        normalization, invalid.loss, invalid.training,
                        invalid.training.seed, false);
        },
        "Early stopping accepted zero patience");
}

void test_regularization_config_validation() {
    Dataset dataset = tiny_dataset(2);
    const std::vector<size_t> training{0};
    const std::vector<size_t> validation{1};
    const NormalizationStats normalization =
        dataset.fit_normalization(training);
    ModelFactory factory;
    Trainer trainer(MPI_COMM_WORLD);

    for (const float invalid_l1 : {-1.0f,
                                   std::numeric_limits<float>::quiet_NaN()}) {
        TrialConfig config = minimal_trial(0.0f);
        config.loss.l1_weight = invalid_l1;
        auto model = factory.build(config, 1, 1, 2, 23, MPI_COMM_WORLD);
        require_throws(
            [&] {
                trainer.fit(*model, dataset, training, dataset, validation,
                            normalization, config.loss, config.training, 23,
                            false);
            },
            "Trainer accepted an invalid L1 weight");
    }

    TrialConfig config = minimal_trial(0.0f);
    config.loss.l2_weight = std::numeric_limits<float>::quiet_NaN();
    auto model = factory.build(config, 1, 1, 2, 23, MPI_COMM_WORLD);
    require_throws(
        [&] {
            trainer.fit(*model, dataset, training, dataset, validation,
                        normalization, config.loss, config.training, 23,
                        false);
        },
        "Trainer accepted an invalid L2 weight");
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

class DiagnosticsFailureRunner final : public TrialRunner {
public:
    FoldMetrics run(const TrialConfig&,
                    const Dataset&,
                    const FoldIndices&,
                    size_t) const override {
        throw DiagnosticsError("simulated recorder failure");
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

    CrossValidator diagnostics_failure(
        dataset, std::make_shared<RandomKFold>(3, true, 2),
        std::make_shared<DiagnosticsFailureRunner>(), MPI_COMM_WORLD, false);
    require_throws(
        [&] { diagnostics_failure.tune(grid); },
        "CrossValidator changed candidate selection after a diagnostics failure");

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

        TrialConfig divergent_diagnostics = minimal_trial();
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        divergent_diagnostics.training.diagnostics.enabled = rank != 0;
        require_throws(
            [&] { validator.evaluate(divergent_diagnostics); },
            "CrossValidator accepted rank-divergent diagnostics configuration");
    }
}

void test_optimizer_recipes() {
    auto sgd_recipe = Recipes::sgd(1e-3f, 0.0f, 0.0f);
    auto sgd_mom_recipe = Recipes::sgd_momentum(1e-3f, 0.9f, 0.0f);
    auto adagrad_recipe = Recipes::adagrad(1e-2f, 1e-8f, 0.0f);
    auto rmsprop_recipe = Recipes::rmsprop(1e-4f, 0.9f, 1e-8f, 0.0f);
    auto adam_recipe = Recipes::adam(1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f);

    auto opt1 = sgd_recipe.build();
    auto opt2 = sgd_mom_recipe.build();
    auto opt3 = adagrad_recipe.build();
    auto opt4 = rmsprop_recipe.build();
    auto opt5 = adam_recipe.build();

    require(opt1 != nullptr, "SGD recipe produced null optimizer");
    require(opt2 != nullptr, "SGD Momentum recipe produced null optimizer");
    require(opt3 != nullptr, "AdaGrad recipe produced null optimizer");
    require(opt4 != nullptr, "RMSprop recipe produced null optimizer");
    require(opt5 != nullptr, "Adam recipe produced null optimizer");

    // Verify fresh instance creation
    auto opt1_second = sgd_recipe.build();
    require(opt1.get() != opt1_second.get(), "Optimizer recipe must construct distinct instances");

    // Verify set_learning_rate and get_learning_rate
    for (auto* opt : {opt1.get(), opt2.get(), opt3.get(), opt4.get(), opt5.get()}) {
        opt->set_learning_rate(0.05f);
        require_close(opt->get_learning_rate(), 0.05f, 1e-6,
                      "Optimizer get_learning_rate must reflect set value");
        require_close(opt->metadata().effective_learning_rate, 0.05f, 1e-6,
                      "Optimizer metadata must reflect updated learning rate");
        require_throws([&] { opt->set_learning_rate(-0.01f); },
                       "Negative learning rate must throw");
        require_throws([&] { opt->set_learning_rate(0.0f); },
                       "Zero learning rate must throw");
        require_throws(
            [&] { opt->set_learning_rate(std::numeric_limits<float>::quiet_NaN()); },
            "NaN learning rate must throw");
    }
}

void test_lr_schedulers() {
    // ConstantLR
    ConstantLR constant_lr(1e-3f);
    require_close(constant_lr.get_rate(0, 100), 1e-3f, 1e-7, "ConstantLR at epoch 0");
    require_close(constant_lr.get_rate(50, 100), 1e-3f, 1e-7, "ConstantLR at epoch 50");
    require_close(constant_lr.get_rate(99, 100), 1e-3f, 1e-7, "ConstantLR at epoch 99");
    require_throws([] { ConstantLR invalid(-1e-3f); }, "ConstantLR negative rate must throw");

    // StepLR
    StepLR step_lr(1e-1f, 15, 0.5f);
    require_close(step_lr.get_rate(0, 100), 0.1f, 1e-7, "StepLR epoch 0");
    require_close(step_lr.get_rate(14, 100), 0.1f, 1e-7, "StepLR epoch 14");
    require_close(step_lr.get_rate(15, 100), 0.05f, 1e-7, "StepLR epoch 15");
    require_close(step_lr.get_rate(29, 100), 0.05f, 1e-7, "StepLR epoch 29");
    require_close(step_lr.get_rate(30, 100), 0.025f, 1e-7, "StepLR epoch 30");
    require_close(step_lr.get_rate(45, 100), 0.0125f, 1e-7, "StepLR epoch 45");

    require_throws([] { StepLR invalid(0.1f, 0, 0.5f); }, "StepLR step_size 0 must throw");
    require_throws([] { StepLR invalid(0.1f, 15, 0.0f); }, "StepLR gamma 0 must throw");
    require_throws([] { StepLR invalid(0.1f, 15, 1.5f); }, "StepLR gamma > 1 must throw");

    // CosineAnnealingLR
    CosineAnnealingLR cosine_lr(1e-5f, 1e-1f);
    require_close(cosine_lr.get_rate(0, 100), 0.1f, 1e-6, "CosineAnnealingLR epoch 0");
    require_close(cosine_lr.get_rate(99, 100), 1e-5f, 1e-7, "CosineAnnealingLR epoch 99");
    const float mid_rate = cosine_lr.get_rate(49, 100);
    require(mid_rate < 0.1f && mid_rate > 1e-5f, "CosineAnnealingLR mid epoch range");
    require_throws([] { CosineAnnealingLR invalid(-1e-5f, 1e-1f); },
                   "CosineAnnealingLR negative min_lr must throw");
    require_throws([] { CosineAnnealingLR invalid(1e-1f, 1e-5f); },
                   "CosineAnnealingLR min_lr > max_lr must throw");

    // WarmupCosineLR
    WarmupCosineLR warmup_lr(1e-5f, 1e-1f, 1e-5f, 5);
    require_close(warmup_lr.get_rate(0, 100), 1e-5f, 1e-7, "WarmupCosineLR epoch 0");
    const float warmup_step2 = warmup_lr.get_rate(2, 100);
    const float expected_step2 = 1e-5f + (2.0f / 5.0f) * (0.1f - 1e-5f);
    require_close(warmup_step2, expected_step2, 1e-6, "WarmupCosineLR epoch 2");
    require_close(warmup_lr.get_rate(5, 100), 0.1f, 1e-6, "WarmupCosineLR peak epoch 5");
    require_close(warmup_lr.get_rate(99, 100), 1e-5f, 1e-7, "WarmupCosineLR epoch 99");

    require_throws([] { WarmupCosineLR invalid(0.0f, 1e-1f, 1e-5f, 5); },
                   "WarmupCosineLR zero start must throw");
    require_throws([] { WarmupCosineLR invalid(1e-1f, 1e-3f, 1e-5f, 5); },
                   "WarmupCosineLR peak < start must throw");
}

void test_lr_scheduler_recipes_and_training() {
    auto c_recipe = Recipes::constant_lr(1e-3f);
    auto s_recipe = Recipes::step_lr(1e-1f, 15, 0.5f);
    auto cos_recipe = Recipes::cosine_lr(1e-5f, 1e-1f);
    auto w_recipe = Recipes::warmup_cosine_lr(1e-5f, 1e-1f, 1e-5f, 5);

    require(c_recipe.has_scheduler() && c_recipe.build() != nullptr, "ConstantLR recipe build");
    require(s_recipe.has_scheduler() && s_recipe.build() != nullptr, "StepLR recipe build");
    require(cos_recipe.has_scheduler() && cos_recipe.build() != nullptr, "CosineLR recipe build");
    require(w_recipe.has_scheduler() && w_recipe.build() != nullptr, "WarmupCosineLR recipe build");

    const Dataset dataset = tiny_dataset(12);
    const auto indices = dataset.all_indices();
    const NormalizationStats normalization = dataset.fit_normalization(indices);

    ModelBlueprint blueprint;
    blueprint.feature_layers = {Recipes::flatten()};
    blueprint.head_layers = {Recipes::dense(4), Recipes::dense(1)};

    TrialConfig trial{"lr-schedule-test", blueprint, Recipes::adam(1e-3f),
                      LossConfig{0.25f}, TrainingConfig{5, 4, 1.0f, 7, false},
                      {}, cos_recipe};

    ModelFactory factory;
    auto model = factory.build(trial, dataset.sdf_height(), dataset.sdf_width(),
                               dataset.scalar_features(), trial.training.seed,
                               MPI_COMM_WORLD);
    const Trainer trainer(MPI_COMM_WORLD);
    const TrainingResult result =
        trainer.fit(*model, dataset, indices, dataset, indices, normalization,
                    trial.loss, trial.training, trial.training.seed, false, {},
                    &trial);

    require(std::isfinite(result.training_objective), "Scheduled training objective must be finite");
    require(std::isfinite(result.validation_mse), "Scheduled validation MSE must be finite");
    require(model->learning_rate() > 0.0f, "Model learning rate must be positive");
}
} // namespace

void test_dropout_layer_behavior() {
    require_throws([] { DropoutLayer invalid(1.0f); },
                   "Dropout rate 1.0 must be rejected.");
    require_throws([] { DropoutLayer invalid(-0.1f); },
                   "Negative dropout rates must be rejected.");
    require_throws(
        [] { Recipes::dropout(std::numeric_limits<float>::quiet_NaN()); },
        "A NaN dropout recipe must be rejected.");

    const std::vector<size_t> shape{64, 32};
    auto input = std::make_shared<Tensor>(shape);
    for (size_t i = 0; i < input->size(); ++i) {
        (*input)[i] = 1.0f;
    }

    // Default context is inference: the layer must be the identity in both
    // directions.
    DropoutLayer inference_layer(0.5f, 21);
    const auto identity_output = inference_layer.forward({input});
    for (size_t i = 0; i < input->size(); ++i) {
        require_close((*identity_output)[i], 1.0f, 0.0,
                      "Inference dropout must be the identity");
    }
    const auto identity_gradients = inference_layer.backward(identity_output);
    require(identity_gradients.size() == 1,
            "Dropout backward must return one gradient tensor.");
    for (size_t i = 0; i < input->size(); ++i) {
        require_close((*identity_gradients[0])[i], 1.0f, 0.0,
                      "Inference dropout backward must be a passthrough");
    }

    const auto wrong_shape_gradient = std::make_shared<Tensor>(
        std::vector<size_t>{32, 64});
    require_throws(
        [&] { inference_layer.backward(wrong_shape_gradient); },
        "Identity dropout backward must reject a shape-mismatched gradient");

    DropoutLayer no_forward_layer(0.5f, 21);
    require_throws(
        [&] { no_forward_layer.backward(identity_output); },
        "Dropout backward must reject a call before forward");

    // Training context: elements are either dropped or scaled by 1/(1-rate),
    // the drop fraction concentrates near the rate, and the same context
    // reproduces the same mask.
    LayerExecutionContext context;
    context.training = true;
    context.stream_seed = 1234;
    context.sample_offset = 0;

    DropoutLayer training_layer(0.5f, 21);
    training_layer.set_execution_context(context);
    const auto first_output = training_layer.forward({input});
    size_t dropped = 0;
    for (size_t i = 0; i < input->size(); ++i) {
        const float value = (*first_output)[i];
        require(value == 0.0f || value == 2.0f,
                "Training dropout must zero or rescale every element.");
        dropped += value == 0.0f ? 1 : 0;
    }
    const double drop_fraction =
        static_cast<double>(dropped) / static_cast<double>(input->size());
    require_close(drop_fraction, 0.5, 0.05,
                  "Dropout must drop close to `rate` of the elements");

    const auto gradient_seed = std::make_shared<Tensor>(shape);
    for (size_t i = 0; i < gradient_seed->size(); ++i) {
        (*gradient_seed)[i] = 1.0f;
    }
    const auto masked_gradients = training_layer.backward(gradient_seed);
    for (size_t i = 0; i < input->size(); ++i) {
        require_close((*masked_gradients[0])[i], (*first_output)[i], 0.0,
                      "Dropout backward must apply the forward mask");
    }

    const auto wrong_mask_shape_gradient = std::make_shared<Tensor>(
        std::vector<size_t>{32, 64});
    training_layer.forward({input});
    require_throws(
        [&] { training_layer.backward(wrong_mask_shape_gradient); },
        "Training dropout backward must reject a shape-mismatched gradient");

    require_throws(
        [&] { training_layer.forward({}); },
        "A failed dropout forward must invalidate the backward cache");
    require_throws(
        [&] { training_layer.backward(gradient_seed); },
        "Dropout backward must reject a stale cache after failed forward");

    training_layer.set_execution_context(context);
    const auto repeated_output = training_layer.forward({input});
    for (size_t i = 0; i < input->size(); ++i) {
        require_close((*repeated_output)[i], (*first_output)[i], 0.0,
                      "The same context must reproduce the same mask");
    }

    // Rank-layout invariance: masking the batch in one piece must equal
    // masking it as two slices whose sample offsets locate them inside the
    // global batch, because that is exactly how MPI ranks see their slices.
    const std::vector<size_t> half_shape{32, 32};
    auto lower_half = std::make_shared<Tensor>(half_shape);
    auto upper_half = std::make_shared<Tensor>(half_shape);
    for (size_t i = 0; i < lower_half->size(); ++i) {
        (*lower_half)[i] = 1.0f;
        (*upper_half)[i] = 1.0f;
    }

    DropoutLayer sliced_layer(0.5f, 21);
    LayerExecutionContext lower_context = context;
    lower_context.sample_offset = 0;
    sliced_layer.set_execution_context(lower_context);
    const auto lower_output = sliced_layer.forward({lower_half});

    LayerExecutionContext upper_context = context;
    upper_context.sample_offset = 32;
    sliced_layer.set_execution_context(upper_context);
    const auto upper_output = sliced_layer.forward({upper_half});

    for (size_t i = 0; i < lower_output->size(); ++i) {
        require_close((*lower_output)[i], (*first_output)[i], 0.0,
                      "Sliced dropout must match the full batch (lower half)");
        require_close((*upper_output)[i],
                      (*first_output)[lower_output->size() + i], 0.0,
                      "Sliced dropout must match the full batch (upper half)");
    }
}

void test_dropout_training_integration() {
    const Dataset dataset = tiny_dataset(12);
    const auto indices = dataset.all_indices();
    const NormalizationStats normalization =
        dataset.fit_normalization(indices);

    const auto make_dropout_trial = [](float rate) {
        ModelBlueprint blueprint;
        blueprint.feature_layers = {Recipes::flatten()};
        blueprint.head_layers = {Recipes::dense(4), Recipes::dropout(rate),
                                 Recipes::dense(1)};
        return TrialConfig{"dropout-test",   blueprint,
                           Recipes::adam(1e-3f), LossConfig{0.25f},
                           TrainingConfig{2, 4, 1.0f, 7, true},
                           {}};
    };

    const auto run_training = [&](const TrialConfig& config) {
        ModelFactory factory;
        auto model = factory.build(config, dataset.sdf_height(),
                                   dataset.sdf_width(),
                                   dataset.scalar_features(),
                                   config.training.seed, MPI_COMM_WORLD);
        const Trainer trainer(MPI_COMM_WORLD);
        return trainer.fit(*model, dataset, indices, dataset, indices,
                           normalization, config.loss, config.training,
                           config.training.seed, false);
    };

    // The dropout path must train to finite metrics and stay deterministic:
    // rebuilding the model from the same seed must reproduce the run exactly.
    const TrainingResult first = run_training(make_dropout_trial(0.35f));
    require(std::isfinite(first.training_objective) &&
                std::isfinite(first.validation_mse),
            "Training with dropout must produce finite metrics.");
    const TrainingResult repeated = run_training(make_dropout_trial(0.35f));
    require_close(repeated.validation_mse, first.validation_mse, 0.0,
                  "Dropout training must be deterministic for a fixed seed");
    require_close(repeated.training_objective, first.training_objective, 0.0,
                  "Dropout training objective must be deterministic");

    // A zero rate must run the identity path, and a positive rate must
    // actually change the optimization trajectory.
    const TrainingResult without = run_training(make_dropout_trial(0.0f));
    require(std::isfinite(without.validation_mse),
            "Zero-rate dropout training must produce finite metrics.");
    require(std::abs(without.validation_mse - first.validation_mse) > 0.0,
            "A positive dropout rate must change the training trajectory.");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    try {
        test_random_kfold();
        test_fold_normalization_and_schema();
        test_physical_simm_scaling();
        test_diagnostics_math_and_activation_capture();
        test_parameter_grid_and_fresh_models();
        test_trainer_with_partial_batches();
        test_historical_range_tail_batching();
        test_early_stopping_policy();
        test_regularization_config_validation();
        test_dropout_layer_behavior();
        test_dropout_training_integration();
        test_generic_checkpoint_round_trip();
        test_training_diagnostics_artifacts();
        test_cross_validator_selection();
        test_optimizer_recipes();
        test_lr_schedulers();
        test_lr_scheduler_recipes_and_training();
        std::cout << "All cross-validation tests passed." << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << std::endl;
        MPI_Finalize();
        return 1;
    }
    MPI_Finalize();
    return 0;
}
