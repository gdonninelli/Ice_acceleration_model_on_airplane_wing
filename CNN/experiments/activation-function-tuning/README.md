# Activation-Function Tuning

## Question

Which activation function gives the lowest physical-unit validation MSE for
the fixed `conv5x5-dense-1024-512-256-128` model?

The experiment evaluates six candidates on the same deterministic five-fold
split:

- `TanhLayer`
- `SigmoidLayer`
- `ReLULayer`
- `LeakyReLULayer(alpha=0.01)`
- `LeakyReLULayer(alpha=0.05)`
- `LeakyReLULayer(alpha=0.1)`

All candidates use the same activation in the convolutional trunk and every
dense hidden layer. The output layer remains linear.

## Fixed Configuration

| Setting | Value |
|---|---|
| Architecture | `conv5x5-dense-1024-512-256-128` |
| Optimizer | Adam |
| Learning rate | `1e-5` |
| Physics weight | `0.1` |
| L1/L2 regularization | `0` / `0` |
| Dropout | `0` |
| Global batch size | `64` |
| Epochs | `100` |
| CV folds | `5` |
| Seed | `42` |

Selection uses the mean physical-unit validation MSE. The untouched test NPZ
is loaded only after the best activation has been selected and is used once
for the final refit/evaluation.

## Build Directly With MPI

From the repository root on the cluster:

```bash
mkdir -p build/experiments
mpicxx -std=c++20 -O3 -ICNN/src \
  -DCNN_SOURCE_REVISION=\"$(git rev-parse --short=12 HEAD)\" \
  CNN/experiments/activation-function-tuning/main.cpp \
  CNN/src/core/*.cpp CNN/src/data/*.cpp CNN/src/layers/*.cpp \
  CNN/src/model/*.cpp CNN/src/optimizers/*.cpp \
  CNN/src/training/*.cpp CNN/src/tuning/*.cpp \
  -o build/experiments/activation-function-tuning
```

The direct commands above build and launch the activation sweep. Account,
partition, modules, and wall time are intentionally left as site-specific
values. The supplied `slurm.sh` is reserved for the configured ordinary CNN
training from `CNN/main.cpp`; use the commands above when submitting this
activation sweep.

## Run

From the repository root:

```bash
mpirun -n 16 build/experiments/activation-function-tuning \
  --train-path dataset/cnn_dataset_train.npz \
  --test-path dataset/cnn_dataset_test.npz \
  --results-dir results/cross_validation/activation-function-tuning \
  --diagnostic
```

`--diagnostic` is the requested spelling; `--diagnostics` is also accepted.
Use `--smoke` for a two-fold, two-epoch validation of the executable and data
paths before submitting the full run.

## Outputs

The experiment writes aggregate files under the selected results directory:

- `fold_results.csv`
- `training_history.csv`
- `summary.txt`

With diagnostics enabled, the standard recorder additionally writes
`<results-dir>/activation-function-tuning/search/` with one directory per
candidate and fold, plus `cv_summary.csv`. Final-refit diagnostics are written
under `.../search/final/`.

## Recorded Results

The completed run used source revision `8aada70b492a`, 64 MPI ranks, five
folds, and 100 epochs. The candidate ranking below is by mean physical-unit
validation MSE; the value after `+/-` is the fold standard deviation.

| Rank | Activation | Mean validation MSE | Fold stddev |
|---:|---|---:|---:|
| 1 | `leakyrelu-alpha-0.05` | **0.00442742** | 0.000865257 |
| 2 | `leakyrelu-alpha-0.01` | 0.00444772 | 0.000876077 |
| 3 | `relu` | 0.00445429 | 0.000875881 |
| 4 | `leakyrelu-alpha-0.1` | 0.00454457 | 0.000965707 |
| 5 | `tanh` | 0.00546562 | 0.00104545 |
| 6 | `sigmoid` | 0.00897140 | 0.00125009 |

## Stability Analysis and Selection

Selection considered both validation performance and training stability. The
stability checks used the five fold diagnostics over all 100 epochs:

| Activation | CV fold stddev | Mean checkpoint fold stddev | Peak gradient norm | Peak weight update ratio | Non-finite metrics |
|---|---:|---:|---:|---:|---:|
| `tanh` | 0.00104545 | 0.00120968 | 111.51 | 3.01e-4 | No |
| `sigmoid` | 0.00125009 | 0.02918695 | 18.22 | 6.15e-4 | No |
| `relu` | 0.000875881 | 0.00109799 | 20.81 | 3.31e-4 | No |
| `leakyrelu-alpha-0.01` | 0.000876077 | 0.00104997 | 20.64 | 4.47e-4 | No |
| `leakyrelu-alpha-0.05` | **0.000865257** | 0.00105385 | **20.14** | 4.50e-4 | No |
| `leakyrelu-alpha-0.1` | 0.000965707 | 0.00109082 | 20.79 | 4.42e-4 | No |

The correct choice is therefore `LeakyReLU(alpha=0.05)`: it has the lowest
mean validation MSE, the smallest final fold-to-fold dispersion, and the
lowest peak gradient among the ReLU-family candidates. Its update ratios stay
small and comparable to the neighboring LeakyReLU settings, indicating that
the result is not caused by an unstable optimizer trajectory. Tanh has a
large transient gradient spike, and Sigmoid has substantially higher
checkpoint variability caused by slow early convergence. All metrics remained
finite, so no candidate exhibited numerical divergence.

The final refit on the complete training set, evaluated once on the untouched
test NPZ, achieved a physical-unit MSE of **0.00273892**. The test result was
not used for selection.

## Plots

Generate the plots from the recorded aggregate CSVs and per-fold diagnostics:

```bash
python3 CNN/experiments/activation-function-tuning/plot_results.py \
    --results-dir results/cross_validation/activation-function-tuning
```

PNG files are written to `CNN/experiments/activation-function-tuning/plots/`:

- `plot1_val_mse_by_activation.png`: mean validation MSE with fold error bars
- `plot2_val_mse_by_fold.png`: validation MSE for every fold and candidate
- `plot3_val_curves_epoch.png`: validation MSE at ten-epoch checkpoints
- `plot4_diagnostics_over_epochs.png`: gradient norms and weight update ratios
