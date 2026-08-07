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
#include <fstream>
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

ModelBlueprint make_blueprint(const std::string& activation, float alpha = 0.05f) {
    ModelBlueprint blueprint;
    blueprint.feature_layers = {
        Recipes::conv2d(8, 5, 5),
        Recipes::activation(activation, alpha),
        Recipes::flatten()};

    blueprint.head_layers = {
        Recipes::dense(128),
        Recipes::activation(activation, alpha),
        Recipes::dense(64),
        Recipes::activation(activation, alpha),
        Recipes::dense(1)};
    return blueprint;
}

void write_csv(const std::string& candidate_name, const CandidateResult& result) {
    std::string path = "results/cross_validation/activation_tuning/" + candidate_name + ".csv";
    std::ofstream f(path);
    if (!f) return;
    f << "candidate,fold,train_mse,val_mse,baseline_mse,epochs\n";
    for (const auto& fold : result.folds) {
        f << candidate_name << "," << fold.fold << "," 
          << fold.training_objective << "," << fold.validation_mse << "," 
          << fold.baseline_mse << "," << kEpochCount << "\n";
    }
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try {
        std::string target_candidate = "";
        if (argc > 1) {
            target_candidate = argv[1];
        }

        Dataset training_dataset("dataset/cnn_dataset_train.npz");

        TrialConfig baseline{
            "activation-tuning",
            make_blueprint("leakyrelu"),
            Recipes::adam(kLearningRate),
            LossConfig{kPhysicsWeight},
            TrainingConfig{kEpochCount, kGlobalBatchSize, 1.0f, kSeed, true},
            {}};

        ParameterGrid grid(baseline);
        std::vector<NamedChoice<ModelBlueprint>> choices = {
             {"relu", make_blueprint("relu")},
             {"tanh", make_blueprint("tanh")},
             {"sigmoid", make_blueprint("sigmoid")},
             {"leakyrelu-0.01", make_blueprint("leakyrelu", 0.01f)},
             {"leakyrelu-0.05", make_blueprint("leakyrelu", 0.05f)},
             {"leakyrelu-0.1", make_blueprint("leakyrelu", 0.1f)},
             {"leakyrelu-0.2", make_blueprint("leakyrelu", 0.2f)},
             {"leakyrelu-0.3", make_blueprint("leakyrelu", 0.3f)}
        };

        if (!target_candidate.empty()) {
            std::vector<NamedChoice<ModelBlueprint>> filtered;
            for (const auto& c : choices) {
                if (c.label == target_candidate) {
                    filtered.push_back(c);
                }
            }
            if (filtered.empty()) {
                if (rank == 0) std::cerr << "Candidate not found!" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            choices = filtered;
        }

        grid.add_choice<ModelBlueprint>(
            "activation-function",
            choices,
            [](TrialConfig& trial, const ModelBlueprint& model) {
                trial.model = model;
            });

        auto splitter = std::make_shared<RandomKFold>(kFoldCount, true, kSeed);
        auto runner = std::make_shared<CNNTrialRunner>(MPI_COMM_WORLD);
        CrossValidator validator(training_dataset, splitter, runner,
                                 MPI_COMM_WORLD, true);

        const SearchResult result = validator.tune(grid);
        
        if (rank == 0) {
            for (const CandidateResult& candidate : result.candidates) {
                if (candidate.success) {
                    write_csv(candidate.config.selected_parameters.at("activation-function"), candidate);
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "Tuning failed on rank " << rank << ": " << error.what()
                  << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
