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

The supplied `slurm.sh` contains the same direct build and the cluster launch
command. Account, partition, modules, and wall time are intentionally left as
site-specific values.

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
