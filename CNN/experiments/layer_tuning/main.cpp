// Layer-architecture cross-validation experiment.
//
// Scientific question: which convolutional/dense topology minimizes the
// physical-unit validation MSE for lift-coefficient prediction? We compare
// Conv2D kernel size, feature-map count, and dense head widths while every
// other hyperparameter (optimizer, learning rate, physics weight, folds,
// epochs, batch size, seed) is held fixed so the comparison isolates topology.
//
// Selection metric: sample-weighted physical-unit validation MSE, minimized by
// CrossValidator. The untouched test NPZ is loaded only after selection.
//
// See CNN/experiments/layer_tuning/README.md and cross_validation.md.

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
constexpr size_t kFoldCount = 5;
constexpr size_t kEpochCount = 100;
constexpr size_t kGlobalBatchSize = 64;
constexpr uint64_t kSeed = 42;
constexpr float kLearningRate = 1e-5f;
constexpr float kPhysicsWeight = 0.25f;

// Feature trunk plus a dense regression head parameterized by the convolution
// output-channel count, the (square) kernel size, and the hidden-layer widths.
// The trunk uses conv2d(out_channels, kernel_size, kernel_size): the third
// argument is the stride, matching the documented experiment template.
ModelBlueprint make_blueprint(int out_channels,
                              int kernel_size,
                              const std::vector<int>& hidden_widths) {
    ModelBlueprint blueprint;
    blueprint.feature_layers = {
        Recipes::conv2d(out_channels, kernel_size, kernel_size),
        Recipes::activation("leakyrelu"),
        Recipes::flatten()};

    for (int width : hidden_widths) {
        blueprint.head_layers.push_back(Recipes::dense(width));
        blueprint.head_layers.push_back(Recipes::activation("leakyrelu"));
    }
    blueprint.head_layers.push_back(Recipes::dense(1));
    return blueprint;
}

void print_candidate(const CandidateResult& candidate) {
    std::cout << "Candidate: " << candidate.config.name << '\n';
    if (!candidate.success) {
        std::cout << "  status: FAILED (" << candidate.error << ")\n";
        return;
    }
    std::cout << "  mean validation MSE (physical): "
              << candidate.mean_validation_mse << " +/- "
              << candidate.validation_stddev << '\n'
              << "  mean training objective:        "
              << candidate.mean_training_objective << '\n';
    for (const auto& [parameter, value] : candidate.config.selected_parameters) {
        std::cout << "  " << parameter << " = " << value << '\n';
    }
    for (const FoldMetrics& fold : candidate.folds) {
        std::cout << "  Fold " << (fold.fold + 1)
                  << " (train=" << fold.training_samples
                  << ", val=" << fold.validation_samples
                  << "): validation_mse=" << fold.validation_mse
                  << " training_objective=" << fold.training_objective << '\n';
        for (const EpochMetrics& point : fold.history) {
            std::cout << "    epoch=" << point.epoch
                      << " train_objective=" << point.training_objective
                      << " validation_mse=" << point.validation_mse << '\n';
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try {
        Dataset training_dataset("dataset/cnn_dataset_train.npz");

        // Fixed baseline: everything except topology stays constant.
        TrialConfig baseline{
            "layer-architecture-search",
            make_blueprint(8, 5, {128, 64}),
            Recipes::adam(kLearningRate),
            LossConfig{kPhysicsWeight},
            TrainingConfig{kEpochCount, kGlobalBatchSize, 1.0f, kSeed, true},
            {}};

        ParameterGrid grid(baseline);
        grid.add_choice<ModelBlueprint>(
            "layer-architecture",
            {{"conv5x5-dense-128-64", make_blueprint(8, 5, {128, 64})},
             {"conv3x3-dense-128-64", make_blueprint(8, 3, {128, 64})},
             {"conv5x5-dense-256-128", make_blueprint(8, 5, {256, 128})},
             {"conv3x3-dense-256-128", make_blueprint(8, 3, {256, 128})},
             {"conv5x5-dense-512-256", make_blueprint(8, 5, {512, 256})},
             {"conv3x3-dense-512-256", make_blueprint(8, 3, {512, 256})},
             {"conv5x5-dense-64-32", make_blueprint(8, 5, {64, 32})},
             // Deeper heads built on the winning width: a third and fourth
             // hidden layer add negligible parameters (the first dense
             // dominates) and probe whether depth helps the representation.
             {"conv5x5-dense-512-256-128",
              make_blueprint(8, 5, {512, 256, 128})},
             {"conv5x5-dense-512-256-128-64",
              make_blueprint(8, 5, {512, 256, 128, 64})},
             // Wider heads beyond the 512-256 winner: does increasing width
             // keep helping, or does it saturate/overfit on 40 samples? The
             // first dense dominates, so each step roughly scales parameters.
             {"conv5x5-dense-768-384", make_blueprint(8, 5, {768, 384})},
             {"conv5x5-dense-1024-512", make_blueprint(8, 5, {1024, 512})},
             {"conv5x5-dense-1536-768", make_blueprint(8, 5, {1536, 768})},
             {"conv5x5-dense-2048-1024", make_blueprint(8, 5, {2048, 1024})},
             // Extrapolating beyond {512, 256, 128, 64}
             {"conv5x5-dense-512-256-128-64-32", make_blueprint(8, 5, {512, 256, 128, 64, 32})}, // Deeper
             {"conv5x5-dense-1024-512-256-128", make_blueprint(8, 5, {1024, 512, 256, 128})}, // Wider, same depth
             {"conv5x5-dense-1024-512-256-128-64", make_blueprint(8, 5, {1024, 512, 256, 128, 64})}, // Deeper and wider
             // Separating depth from width
             {"conv5x5-dense-512-64", make_blueprint(8, 5, {512, 64})},
             {"conv5x5-dense-256-128-64-32", make_blueprint(8, 5, {256, 128, 64, 32})}},
            [](TrialConfig& trial, const ModelBlueprint& model) {
                trial.model = model;
            });

        auto splitter =
            std::make_shared<RandomKFold>(kFoldCount, true, kSeed);
        auto runner = std::make_shared<CNNTrialRunner>(MPI_COMM_WORLD);
        CrossValidator validator(training_dataset, splitter, runner,
                                 MPI_COMM_WORLD, true);

        const SearchResult result = validator.tune(grid);
        const CandidateResult& best = result.best();

        if (rank == 0) {
            std::cout << "\n==================== SEARCH RESULTS "
                         "====================\n";
            for (const CandidateResult& candidate : result.candidates) {
                print_candidate(candidate);
                std::cout << '\n';
            }
            std::cout << "-------------------- WINNER "
                         "--------------------\n"
                      << "Best candidate: " << best.config.name << '\n'
                      << "Validation physical MSE: "
                      << best.mean_validation_mse << " +/- "
                      << best.validation_stddev << '\n';
            for (const auto& [parameter, value] :
                 best.config.selected_parameters) {
                std::cout << "  " << parameter << ": " << value << '\n';
            }
            std::cout << std::flush;
        }

        // Selection is complete. Only now touch the held-out test set: refit the
        // winning architecture on the full training NPZ and evaluate once.
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
            rank == 0);

        if (rank == 0) {
            std::cout << "\n==================== FINAL TEST "
                         "====================\n"
                      << "Refit architecture: " << best.config.name << '\n'
                      << "Final untouched-test physical MSE: "
                      << final_result.validation_mse << '\n'
                      << std::flush;
        }
    } catch (const std::exception& error) {
        std::cerr << "Tuning failed on rank " << rank << ": " << error.what()
                  << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
