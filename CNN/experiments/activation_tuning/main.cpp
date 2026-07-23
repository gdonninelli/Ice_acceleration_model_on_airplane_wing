// Activation-function tuning experiment.
//
// Compares relu, leakyrelu, tanh, and sigmoid activations using the shared
// CrossValidator / ParameterGrid architecture described in cross_validation.md.
// The activation choice is applied uniformly to every activation site in the
// network (feature trunk and dense head), so each candidate differs only in the
// non-linearity used. Candidate selection minimises physical-unit validation
// MSE; the untouched test set is loaded only once, after the winner is chosen.

#include "data/Dataset.hpp"
#include "model/ModelFactory.hpp"
#include "training/Trainer.hpp"
#include "tuning/CrossValidator.hpp"
#include "tuning/SearchSpace.hpp"
#include "tuning/TrialConfig.hpp"
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

namespace {
// Baseline experiment budget. Keep the fold seed and count fixed across every
// candidate so activation scores remain directly comparable.
constexpr size_t kFoldCount = 5;
constexpr size_t kEpochCount = 100;
constexpr size_t kGlobalBatchSize = 256;
constexpr uint64_t kSeed = 42;
constexpr float kLearningRate = 1e-5f;
constexpr float kPhysicsWeight = 0.25f;

// Builds the fixed baseline topology with a single configurable activation:
//   Feature: Conv2D(8, 5x5) -> Activation -> Flatten
//   Head:    Dense(128) -> Activation -> Dense(64) -> Activation -> Dense(1)
// Only the non-linearity varies between candidates.
ModelBlueprint make_blueprint(const std::string& activation) {
    ModelBlueprint blueprint;
    blueprint.feature_layers = {
        Recipes::conv2d(8, 5, 5),
        Recipes::activation(activation),
        Recipes::flatten()};

    blueprint.head_layers = {
        Recipes::dense(128),
        Recipes::activation(activation),
        Recipes::dense(64),
        Recipes::activation(activation),
        Recipes::dense(1)};
    return blueprint;
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try {
        // Cross-validation uses the training NPZ only.
        Dataset training_dataset("dataset/cnn_dataset_train.npz");

        TrialConfig baseline{
            "activation-tuning",
            make_blueprint("leakyrelu"),
            Recipes::adam(kLearningRate),
            LossConfig{kPhysicsWeight},
            TrainingConfig{kEpochCount, kGlobalBatchSize, 1.0f, kSeed, true},
            {}};

        ParameterGrid grid(baseline);
        grid.add_choice<ModelBlueprint>(
            "activation-function",
            {{"relu", make_blueprint("relu")},
             {"leakyrelu", make_blueprint("leakyrelu")},
             {"tanh", make_blueprint("tanh")},
             {"sigmoid", make_blueprint("sigmoid")}},
            [](TrialConfig& trial, const ModelBlueprint& model) {
                trial.model = model;
            });

        auto splitter = std::make_shared<RandomKFold>(kFoldCount, true, kSeed);
        auto runner = std::make_shared<CNNTrialRunner>(MPI_COMM_WORLD);
        CrossValidator validator(training_dataset, splitter, runner,
                                 MPI_COMM_WORLD, true);

        const SearchResult result = validator.tune(grid);
        const CandidateResult& best = result.best();

        if (rank == 0) {
            std::cout << "\n===== Activation search summary =====\n";
            for (const CandidateResult& candidate : result.candidates) {
                std::cout << candidate.config.name << '\n';
                if (!candidate.success) {
                    std::cout << "  status: FAILED (" << candidate.error
                              << ")\n";
                    continue;
                }
                std::cout << "  mean validation MSE: "
                          << candidate.mean_validation_mse << " +/- "
                          << candidate.validation_stddev << '\n'
                          << "  mean training objective: "
                          << candidate.mean_training_objective << '\n';
            }

            std::cout << "\n===== Best candidate =====\n"
                      << "Name: " << best.config.name << '\n'
                      << "Validation physical MSE: " << best.mean_validation_mse
                      << " +/- " << best.validation_stddev << '\n';
            for (const auto& [parameter, value] :
                 best.config.selected_parameters) {
                std::cout << "  " << parameter << ": " << value << '\n';
            }
            for (const FoldMetrics& fold : best.folds) {
                std::cout << "Fold " << (fold.fold + 1)
                          << " validation MSE=" << fold.validation_mse
                          << " (train=" << fold.training_samples
                          << ", val=" << fold.validation_samples << ")\n";
                for (const EpochMetrics& point : fold.history) {
                    std::cout << "  epoch=" << point.epoch
                              << " train_objective=" << point.training_objective
                              << " validation_mse=" << point.validation_mse
                              << '\n';
                }
            }
        }

        // Only now load the untouched test set and refit the selected candidate
        // on the complete training dataset for a single unbiased estimate.
        Dataset test_dataset("dataset/cnn_dataset_test.npz");
        const auto training_indices = training_dataset.all_indices();
        const auto test_indices = test_dataset.all_indices();
        const NormalizationStats normalization =
            training_dataset.fit_normalization(training_indices);

        ModelFactory factory;
        auto final_model = factory.build(
            best.config,
            training_dataset.sdf_height(),
            training_dataset.sdf_width(),
            training_dataset.scalar_features(),
            best.config.training.seed,
            MPI_COMM_WORLD);

        Trainer trainer(MPI_COMM_WORLD);
        const TrainingResult final_result = trainer.fit(
            *final_model,
            training_dataset,
            training_indices,
            test_dataset,
            test_indices,
            normalization,
            best.config.loss,
            best.config.training,
            best.config.training.seed,
            true);

        if (rank == 0) {
            std::cout << "\n===== Final unbiased estimate =====\n"
                      << "Winning activation: " << best.config.name << '\n'
                      << "Final untouched-test physical MSE: "
                      << final_result.validation_mse << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Tuning failed on rank " << rank << ": " << error.what()
                  << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
