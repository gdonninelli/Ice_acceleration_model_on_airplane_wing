# Cross-Validation and Hyperparameter Tuning

This document describes the typed C++ tuning architecture built around a single `CrossValidator`. The validator is independent of concrete layers, optimizers, and search parameters: developers provide complete `TrialConfig` candidates, and every candidate/fold creates fresh runtime state.

It covers:

- Running K-fold evaluation from the existing executable
- Creating a separate tuning program without changing `CNN/main.cpp`
- Preparing train and untouched-test datasets
- Defining model and optimizer choices with `ParameterGrid`
- Compiling and running a tuning experiment with MPI
- Reading, recording, and validating results
- Reading per-fold training history at fixed 10-epoch checkpoints
- Attaching future layers and optimizers through recipes

## Architecture

```text
ParameterGrid -> TrialConfig candidates
                       |
                       v
                CrossValidator
              /        |        \
       FoldSplitter  TrialRunner  SearchResult
                        |
                 ModelFactory + Trainer
                 /        |        \
          ModelBlueprint  Loss  OptimizerRecipe
```

The main contracts are located under `CNN/src/tuning/`:

| Contract | Responsibility |
|---|---|
| `TrialConfig` | Complete model, optimizer, loss, and training configuration |
| `LayerRecipe` | Shape inference and fresh construction of one layer |
| `OptimizerRecipe` | Fresh construction of one optimizer |
| `ModelBlueprint` | Ordered feature trunk, scalar fusion, and dense head |
| `ParameterGrid` | Typed Cartesian product of named parameter choices |
| `RandomKFold` | Deterministic shuffled sample-level folds |
| `TrialRunner` | Executes one candidate on one fold |
| `CrossValidator` | Evaluates candidates and selects the minimum physical MSE |
| `EpochMetrics` | One fixed 10-epoch training-objective and validation-MSE checkpoint |
| `FoldMetrics` | Final fold metrics plus that fold's epoch history |
| `SearchResult` | Candidate configurations, fold metrics, means, deviations, and winner |

Runtime models are non-copyable. Each fold receives newly initialized layers and a newly constructed optimizer, preventing weights, activation caches, and optimizer moments from leaking between folds.

Recipe builders return `std::unique_ptr`, so extension code cannot accidentally share a runtime layer or optimizer between factory builds. Immutable recipes themselves remain safely copyable inside search candidates.

## Data and Scoring

`Dataset` loads raw arrays and follows this schema:

```text
X_sdf:     [sample, 150, 150]
X_scalars: [sample, 2] = [Reynolds number, AoA in degrees]
Y_cl:      [sample]
```

For every fold, normalization is fitted on the training indices and then applied unchanged to validation indices. AoA is read from column 1 and converted from degrees to radians before SIMM is evaluated.

The data term is computed in standardized target space. The physical relation `2*pi*alpha` is transformed into that same fold-specific target space before the SIMM physics residual is evaluated, keeping the physics weight comparable across folds. Candidate selection always uses physical-unit validation MSE, so changing the SIMM physics weight does not change the comparison metric itself.

Cross-validation uses only the training NPZ. `cnn_dataset_test.npz` remains untouched until the winning candidate is rebuilt and trained on the complete training dataset.

`RandomKFold` intentionally splits individual samples, as configured for this project. This measures interpolation when observations from the same airfoil may occur in multiple folds. For unseen-airfoil generalization, provide a metadata-aware `FoldSplitter` that keeps each airfoil entirely within one fold; `CrossValidator` itself does not need to change.

## Recommended Project Layout

Do not replace the production entry point in `CNN/main.cpp` for every tuning experiment. Create one directory per experiment under `CNN/experiments/`:

```text
.
├── CNN/
│   ├── main.cpp                         # Normal training and K-fold evaluation
│   ├── experiments/
│   │   └── kernel_activation/
│   │       ├── main.cpp                 # Experiment-specific search space
│   │       └── README.md                # Question, ranges, budget, conclusions
│   └── src/
│       ├── data/
│       ├── model/
│       ├── training/
│       └── tuning/
├── dataset/
│   ├── cnn_dataset_train.npz            # Used for folds and final refit
│   └── cnn_dataset_test.npz             # Used once after model selection
├── results/
│   └── cross_validation/
│       └── kernel_activation/
│           └── run.log                  # Generated output, normally not committed
└── cross_validation.md
```

Commit the experiment's `main.cpp` and short `README.md` when the experiment must be reproducible. Keep large logs, checkpoints, and generated datasets out of Git unless the project explicitly requires them.

If this layout is adopted, add `results/` to `.gitignore` or to your local Git exclude file before running experiments.

All documented commands assume they are executed from the repository root. This keeps dataset paths consistently relative to `dataset/`.

## Input Data Location

Place input data at:

```text
dataset/cnn_dataset_train.npz
dataset/cnn_dataset_test.npz
```

The training NPZ is the only file passed to `CrossValidator`. The test NPZ must not be loaded while comparing candidates. Use it once after selecting a configuration and refitting that configuration on every training sample.

For a custom location, construct `Dataset` with the explicit path or use the executable options:

```bash
--train-path /path/to/train.npz --test-path /path/to/test.npz
```

## Build

With CMake:

```bash
cmake -S CNN -B build/CNN -DCMAKE_BUILD_TYPE=Release
cmake --build build/CNN --parallel
ctest --test-dir build/CNN --output-on-failure
```

Without CMake:

```bash
cd CNN
mpicxx -std=c++20 -O3 -Isrc \
  main.cpp src/core/*.cpp src/data/*.cpp src/layers/*.cpp \
  src/model/*.cpp src/optimizers/*.cpp src/training/*.cpp \
  src/tuning/*.cpp -o cnn_executable
```

## Run K-Fold Evaluation

From the repository root:

```bash
mpirun -n 4 ./CNN/cnn_executable \
  --cross-validate \
  --folds 5 \
  --epochs 100 \
  --batch-size 256 \
  --seed 42
```

`--batch-size` is the global MPI batch size. It does not scale with the number of ranks. Partial final batches are processed, and gradient aggregation is weighted by each rank's actual sample count.

The executable evaluates the currently configured model across the requested folds. It does not create pooling layers, alternative optimizers, or a predefined hyperparameter search. Those remain extension choices supplied through `TrialConfig` and `ParameterGrid`.

For a short smoke run:

```bash
mpirun -n 2 ./CNN/cnn_executable --cross-validate --folds 2 --epochs 1
```

## Prepare a Tuning Experiment

Use this process before launching an expensive search:

1. Define the scientific question, such as kernel size versus activation choice.
2. Choose one fixed selection metric. This implementation minimizes physical-unit validation MSE.
3. Keep `cnn_dataset_test.npz` outside the tuning loop.
4. Start from a known baseline `TrialConfig`.
5. Add only justified candidate choices to `ParameterGrid`.
6. Estimate the total runs as `candidate count * fold count`.
7. Run a two-fold, one-epoch smoke test.
8. Run the complete search only after the smoke test succeeds.
9. Refit the winning configuration on the complete training NPZ.
10. Evaluate the untouched test NPZ once and record the result.

Avoid changing the fold seed between candidates. Every candidate must be evaluated on the same folds for scores to be comparable.

## Experiment `main.cpp` Template

Create `CNN/experiments/kernel_activation/main.cpp` with a structure like this:

```cpp
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

ModelBlueprint make_blueprint(int kernel_size,
                              const std::string& activation,
                              const std::vector<int>& hidden_widths) {
    ModelBlueprint blueprint;
    blueprint.feature_layers = {
        Recipes::conv2d(8, kernel_size, kernel_size),
        Recipes::activation(activation),
        Recipes::flatten()};

    for (int width : hidden_widths) {
        blueprint.head_layers.push_back(Recipes::dense(width));
        blueprint.head_layers.push_back(Recipes::activation(activation));
    }
    blueprint.head_layers.push_back(Recipes::dense(1));
    return blueprint;
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try {
        Dataset training_dataset("dataset/cnn_dataset_train.npz");

        TrialConfig baseline{
            "kernel-activation-search",
            make_blueprint(5, "leakyrelu", {128, 64}),
            Recipes::adam(1e-5f),
            LossConfig{0.25f},
            TrainingConfig{kEpochCount, 256, 1.0f, 42, true},
            {}};

        ParameterGrid grid(baseline);
        grid.add_choice<ModelBlueprint>(
            "architecture",
            {{"kernel3-relu", make_blueprint(3, "relu", {128, 64})},
             {"kernel5-leaky", make_blueprint(5, "leakyrelu", {128, 64})},
             {"kernel5-tanh", make_blueprint(5, "tanh", {128, 64})}},
            [](TrialConfig& trial, const ModelBlueprint& model) {
                trial.model = model;
            });

        grid.add_choice<OptimizerRecipe>(
            "adam-learning-rate",
            {{"1e-5", Recipes::adam(1e-5f)},
             {"1e-4", Recipes::adam(1e-4f)}},
            [](TrialConfig& trial, const OptimizerRecipe& optimizer) {
                trial.optimizer = optimizer;
            });

        auto splitter = std::make_shared<RandomKFold>(kFoldCount, true, 42);
        auto runner = std::make_shared<CNNTrialRunner>(MPI_COMM_WORLD);
        CrossValidator validator(training_dataset, splitter, runner,
                                 MPI_COMM_WORLD, true);

        const SearchResult result = validator.tune(grid);
        const CandidateResult& best = result.best();

        if (rank == 0) {
            std::cout << "Best candidate: " << best.config.name << '\n'
                      << "Validation physical MSE: "
                      << best.mean_validation_mse << " +/- "
                      << best.validation_stddev << '\n';
            for (const auto& [parameter, value] :
                 best.config.selected_parameters) {
                std::cout << "  " << parameter << ": " << value << '\n';
            }
            for (const FoldMetrics& fold : best.folds) {
                std::cout << "Fold " << (fold.fold + 1) << " history:\n";
                for (const EpochMetrics& point : fold.history) {
                    std::cout << "  epoch=" << point.epoch
                              << " train_objective="
                              << point.training_objective
                              << " validation_mse="
                              << point.validation_mse << '\n';
                }
            }
        }

        // Only now load the untouched test set and refit the selected candidate.
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
            std::cout << "Final untouched-test physical MSE: "
                      << final_result.validation_mse << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Tuning failed on rank " << rank << ": "
                  << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
```

This example uses only components already implemented in the repository. A future component is attached by adding its recipe as another typed choice; `CrossValidator` remains unchanged.

## Build a Tuning Program

From the repository root:

```bash
mkdir -p build/experiments

mpicxx -std=c++20 -O3 -ICNN/src \
  CNN/experiments/kernel_activation/main.cpp \
  CNN/src/core/*.cpp CNN/src/data/*.cpp CNN/src/layers/*.cpp \
  CNN/src/model/*.cpp CNN/src/optimizers/*.cpp \
  CNN/src/training/*.cpp CNN/src/tuning/*.cpp \
  -o build/experiments/kernel_activation
```

Create the output directory:

```bash
mkdir -p results/cross_validation/kernel_activation
```

For a smoke test, temporarily set `kFoldCount = 2` and `kEpochCount = 1`, rebuild, and run:

```bash
mpirun -n 2 build/experiments/kernel_activation \
  | tee results/cross_validation/kernel_activation/smoke.log
```

Restore the intended fold and epoch counts, rebuild, and write the full run to a separate log after the smoke test succeeds.

The current API returns results in memory through `SearchResult` and prints progress to standard output. It does not automatically write JSON, CSV, or checkpoints. The experiment program is responsible for any additional serialization.

## What to Record

Record at least the following information for every completed search:

| Item | Source |
|---|---|
| Experiment name | `TrialConfig::name` |
| Candidate values | `CandidateResult::config.selected_parameters` |
| Fold seed and count | `RandomKFold` construction |
| Epochs and global batch size | `TrainingConfig` |
| Training objective | `FoldMetrics::training_objective` |
| Validation score | `FoldMetrics::validation_mse` |
| Aggregate score | `CandidateResult::mean_validation_mse` |
| Fold variability | `CandidateResult::validation_stddev` |
| Per-fold 10-epoch history | `FoldMetrics::history` |
| MPI process count | Launch command or scheduler log |
| Source revision | Git commit hash used for the run |

Do not select a model from test-set performance. Selection belongs to cross-validation; the test set is only for the final unbiased estimate.

## Ordinary Training

The non-tuning path uses the same `ModelFactory`, `Trainer`, fold-safe preprocessing, and physical metric:

```bash
mpirun -n 4 ./CNN/cnn_executable \
  --activation leakyrelu \
  --alpha 0.05 \
  --learning-rate 1e-5 \
  --epochs 100 \
  --batch-size 256
```

Run `./CNN/cnn_executable --help` for all options.

## Typed API

Create a concrete base trial and add typed choices with setters:

```cpp
TrialConfig base{
    "experiment",
    blueprint,
    Recipes::adam(1e-5f),
    LossConfig{0.25f},
    TrainingConfig{100, 256, 1.0f, 42, true},
    {}};

ParameterGrid grid(base);

grid.add_choice<OptimizerRecipe>(
    "optimizer",
    {{"adam-1e-5", Recipes::adam(1e-5f)},
     {"adam-1e-4", Recipes::adam(1e-4f)}},
    [](TrialConfig& trial, const OptimizerRecipe& optimizer) {
        trial.optimizer = optimizer;
    });

auto splitter = std::make_shared<RandomKFold>(5, true, 42);
auto runner = std::make_shared<CNNTrialRunner>(MPI_COMM_WORLD);
CrossValidator validator(dataset, splitter, runner, MPI_COMM_WORLD);

SearchResult result = validator.tune(grid);
const CandidateResult& best = result.best();
```

`ParameterGrid` accepts any copyable C++ value. A choice can replace one optimizer, one complete `ModelBlueprint`, a loss setting, or any other field in `TrialConfig`. `CrossValidator` does not require a new method for each parameter category.

## Define a Topology

`ModelBlueprint` deliberately models the network already used by this project: a single-input feature trunk, optional scalar concatenation, and a sequential regression head.

```cpp
ModelBlueprint model;
model.feature_layers = {
    Recipes::conv2d(8, 3, 2),
    Recipes::activation("relu"),
    Recipes::flatten()};

model.head_layers = {
    Recipes::dense(128),
    Recipes::activation("relu"),
    Recipes::dense(64),
    Recipes::activation("relu"),
    Recipes::dense(1)};
```

Shape inference runs before training. Dense input widths are derived automatically, and invalid configured layer combinations fail before the expensive fold run begins.

## Add a Layer

Implement `Layer::forward()` and `Layer::backward()`. Parameter-free layers need nothing else because `Layer::parameters()` defaults to an empty collection.

Learnable layers override `parameters()`:

```cpp
std::vector<LayerParameter> CustomLayer::parameters() {
    return {{"weights", weights}, {"biases", biases}};
}
```

Those parameters are then automatically included in:

- Gradient zeroing and clipping
- MPI gradient synchronization
- Optimizer updates
- Initial weight broadcast
- Weight export and import

Expose the layer through a typed recipe containing shape inference and a builder:

```cpp
LayerRecipe custom_layer(CustomConfig config) {
    return LayerRecipe(
        "custom-layer",
        [config](const TensorShape& input) {
            return infer_custom_shape(input, config);
        },
        [config](const TensorShape& input, uint64_t seed) {
            return std::make_unique<CustomLayer>(input, config, seed);
        });
}
```

The recipe can immediately be placed in a blueprint or supplied as a `ParameterGrid` choice. For example, a developer can implement a pooling layer and its recipe without changing `CrossValidator`.

## Add an Optimizer

Implement `Optimizer::apply_gradients()` and expose a recipe:

```cpp
OptimizerRecipe rmsprop(RMSPropConfig config) {
    return OptimizerRecipe("rmsprop", [config] {
        return std::make_unique<RMSPropOptimizer>(config);
    });
}
```

The recipe must construct a new optimizer every time. Optimizer objects must never be shared across candidates or folds.

## Results

`SearchResult::candidates` retains successful and failed candidates. A successful `CandidateResult` includes:

- The resolved `TrialConfig` and selected parameter labels
- Per-fold training objective and physical validation MSE
- Sample-weighted mean validation MSE
- Sample-weighted validation standard deviation
- Training objective and validation MSE history for every fold

`SearchResult::best()` returns the successful candidate with the lowest mean physical validation MSE. Ties retain deterministic grid order.

## Epoch History

History collection uses the compile-time constant:

```cpp
Trainer::kHistoryIntervalEpochs == 10
```

It is intentionally not part of `TrainingConfig` and cannot be changed through `TrialConfig` or command-line parameters.

At epochs 10, 20, 30, and so on, `Trainer` records:

```cpp
struct EpochMetrics {
    size_t epoch;
    double training_objective; // SIMM objective for that training epoch
    double validation_mse;     // physical-unit MSE on that fold's validation set
};
```

Each fold stores its own points in:

```cpp
FoldMetrics::history
```

For 100 epochs, every fold contains 10 history entries. For 25 epochs, every fold contains entries for epochs 10 and 20. The final epoch 25 validation MSE is still calculated and used for candidate scoring, but it is not added to the fixed 10-epoch history. Runs shorter than 10 epochs have an empty history while still producing a final validation MSE.

Access the history with:

```cpp
for (const CandidateResult& candidate : result.candidates) {
    for (const FoldMetrics& fold : candidate.folds) {
        for (const EpochMetrics& point : fold.history) {
            // Save or plot point.epoch, point.training_objective,
            // and point.validation_mse.
        }
    }
}
```

Validation requires an additional pass over the fold's validation samples at every history checkpoint. The final scoring pass is reused when the final epoch is divisible by 10.

## MPI Execution

All ranks evaluate the same candidate and fold. One validated fold plan is generated per `tune()` call and reused for every candidate. Candidate descriptions, fold plans, failures, and returned metrics are checked across ranks before candidate selection can continue.

`MPI_Comm` is injected into `CrossValidator`, `CNNTrialRunner`, `Trainer`, and `CNNModel`; collectives are no longer bound internally to `MPI_COMM_WORLD`. Custom `TrialRunner` implementations must not perform unmatched collectives internally, but exceptions returned from a runner are reconciled before the validator advances to another candidate.

Parallel candidate groups are not implemented. They can be added later by creating subcommunicators with `MPI_Comm_split` and constructing one validator per communicator.

## Tests

`CNN/tests/test_cross_validation.cpp` covers:

- Deterministic, disjoint, complete random folds
- Fold-only normalization and scalar schema
- Physical SIMM forward and backward scaling
- Cartesian parameter-grid expansion
- Fresh model state per factory build
- Partial global batches and MPI-weighted training
- Fixed 10-epoch history retention for every fold
- Best-candidate selection through a fake trial runner

The test executable is safe to run with one or multiple ranks:

```bash
ctest --test-dir build/CNN --output-on-failure
mpirun -n 2 ./build/CNN/cnn_tests
```
