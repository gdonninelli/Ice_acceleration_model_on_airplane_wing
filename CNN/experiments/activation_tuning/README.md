# Activation-Function Tuning Experiment

> ⚠️ **Risultati non disponibili.** Questo esperimento è stato eseguito originariamente su
> un dataset che si è rivelato invalido: i file `.npz` committati nel repository
> contenevano rumore gaussiano anziché le Signed Distance Function dei profili alari
> (autocorrelazione lag-1 ≈ 0.0001 contro ≈ 0.999 attesa; nessuno dei target presente nei
> summary CFD). Tutte le misure e le conclusioni prodotte in quella sede sono state rimosse.
> L'infrastruttura dell'esperimento è valida e riutilizzabile. Le misure verranno rifatte
> sul dataset rigenerato (2142 campioni, 5 profili, verificato fisicamente: pendenza
> C_l vs α = 0.111/grado contro 0.1097 teorico).

## Question

Which activation function minimises the physical-unit validation MSE for the
baseline airfoil-coefficient CNN? This experiment holds every other
hyperparameter fixed and varies only the non-linearity.

## Search Space

A single search axis, `activation-function`, with four candidates. The chosen
activation is applied uniformly at every activation site (feature trunk and
dense head), so candidates differ only in the non-linearity.

| Candidate   | Activation             |
|-------------|------------------------|
| `relu`      | ReLU                   |
| `leakyrelu` | Leaky ReLU (α = 0.05)  |
| `tanh`      | Hyperbolic tangent     |
| `sigmoid`   | Logistic sigmoid       |

Total runs = 4 candidates × 5 folds = **20 fold evaluations**.

## Fixed Baseline (`TrialConfig`)

| Component     | Setting                                                        |
|---------------|----------------------------------------------------------------|
| Feature trunk | `Conv2D(8, 5×5)` → `Activation` → `Flatten`                     |
| Head          | `Dense(128)` → `Activation` → `Dense(64)` → `Activation` → `Dense(1)` |
| Optimizer     | Adam, learning rate `1e-5`                                      |
| Loss          | SIMM physics loss, weight α = `0.25`                            |
| Folds         | 5 (`RandomKFold`, shuffle, seed 42)                            |
| Epochs        | 100                                                             |
| Global batch  | 256 (MPI-global, not per-rank)                                 |
| Seed          | 42                                                              |

The fold seed and count are held fixed across every candidate so scores are
directly comparable. Candidate selection minimises **physical-unit validation
MSE** (`CandidateResult::mean_validation_mse`), independent of the SIMM physics
weight.

## Selection Metric and Test Set

Cross-validation uses `dataset/cnn_dataset_train.npz` only. After the winning
activation is selected, the model is refit on the complete training NPZ and
evaluated exactly once on the untouched `dataset/cnn_dataset_test.npz` to
produce a single unbiased physical-MSE estimate. The test set is never consulted
during candidate comparison.

## Build

From the repository root:

```bash
mkdir -p build/experiments

mpicxx -std=c++20 -O3 -ICNN/src \
  CNN/experiments/activation_tuning/main.cpp \
  CNN/src/core/*.cpp CNN/src/data/*.cpp CNN/src/layers/*.cpp \
  CNN/src/model/*.cpp CNN/src/optimizers/*.cpp \
  CNN/src/training/*.cpp CNN/src/tuning/*.cpp \
  -o build/experiments/activation_tuning
```

## Run

Prepare the output directory and run the full experiment:

```bash
mkdir -p results/cross_validation/activation_tuning

mpirun -n 4 build/experiments/activation_tuning \
  | tee results/cross_validation/activation_tuning/run.log
```

`-n 4` sets the number of MPI ranks; the global batch size (256) does not scale
with rank count. Rank 0 prints the per-candidate scores, the best candidate's
per-fold validation MSE and 10-epoch history, and the final untouched-test MSE.

### Smoke test

Before the full run, verify the build and MPI runtime with a short pass. Reduce
`kFoldCount` to `2` and `kEpochCount` to `1` in `main.cpp`, rebuild, then:

```bash
mpirun -n 2 build/experiments/activation_tuning \
  | tee results/cross_validation/activation_tuning/smoke.log
```

Restore `kFoldCount = 5` and `kEpochCount = 100` and rebuild before the real run.

## Requirements

- An MPI toolchain (`mpicxx` / `mpirun`, e.g. Open MPI or MPICH).
- The dataset NPZ files present at `dataset/cnn_dataset_train.npz` and
  `dataset/cnn_dataset_test.npz` (these are git-ignored and generated
  separately).

## What to Record

Per the checklist in `cross_validation.md`: experiment name, candidate values,
fold seed/count, epochs and global batch size, per-fold training objective and
validation MSE, aggregate mean and standard deviation, the per-fold 10-epoch
history, the MPI process count, and the Git commit hash used for the run.
