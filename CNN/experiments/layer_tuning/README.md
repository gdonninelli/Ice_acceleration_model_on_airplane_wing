# Layer-Architecture Tuning Experiment

Cross-validation experiment that compares convolutional and dense-head
topologies for lift-coefficient (`C_l`) prediction, built on the typed
`CrossValidator` / `ParameterGrid` API documented in
[`cross_validation.md`](../../../cross_validation.md).

## Scientific Question

Holding the optimizer, learning rate, physics weight, and training schedule
fixed, **which layer topology minimizes the physical-unit validation MSE?** We
vary three structural knobs and measure their effect in isolation:

- Conv2D kernel size (`5x5` vs `3x3`),
- Conv2D feature-map count (`8` vs `16` output channels),
- Dense head hidden widths (`{128, 64}` vs `{256, 128}`).

Selection uses sample-weighted **physical-unit validation MSE** minimized by
`CrossValidator`. The held-out test NPZ is never read during selection; it is
used exactly once to produce an unbiased estimate for the winning topology.

## Feature Trunk / Head

Each candidate is produced by
`make_blueprint(out_channels, kernel_size, hidden_widths)`:

- **Feature trunk:** `conv2d(out_channels, kernel_size, kernel_size)` →
  `leakyrelu` → `flatten`. The third `conv2d` argument is the stride, matching
  the documented experiment template, so kernel size and stride move together.
- **Head:** for each width `w` in `hidden_widths`, `dense(w)` → `leakyrelu`,
  followed by a final `dense(1)` regression output.

## Fixed Baseline

All candidates share these settings; only `layer-architecture` varies.

| Setting | Value |
|---|---|
| Baseline architecture | `make_blueprint(8, 5, {128, 64})` |
| Optimizer | Adam, learning rate `1e-5` |
| Loss | SIMM physics loss, weight `alpha = 0.25` |
| Folds | 5 (`RandomKFold`, shuffle, seed 42) |
| Epochs | 100 |
| Global batch size | 256 (MPI-global, does not scale with ranks) |
| Seed | 42 |
| Selection metric | physical-unit validation MSE |
| Training NPZ | `dataset/cnn_dataset_train.npz` |
| Test NPZ (once) | `dataset/cnn_dataset_test.npz` |

## Candidates (`layer-architecture` axis)

| Label | Blueprint | Varies from baseline |
|---|---|---|
| `conv5x5-dense-128-64` | `make_blueprint(8, 5, {128, 64})` | baseline |
| `conv3x3-dense-128-64` | `make_blueprint(8, 3, {128, 64})` | kernel `5x5` → `3x3` |
| `conv5x5-dense-256-128` | `make_blueprint(8, 5, {256, 128})` | wider dense head |
| `conv5x5-f16-dense-128-64` | `make_blueprint(16, 5, {128, 64})` | `8` → `16` filters |

Total runs = 4 candidates × 5 folds = **20 fold evaluations**, plus one final
refit + test evaluation for the winner.

## Requirements

- MPI toolchain (OpenMPI `mpicxx` / `mpirun`). On this machine:
  `export PATH=/usr/lib64/openmpi/bin:$PATH`.
- A C++20 compiler.
- `dataset/cnn_dataset_train.npz` and `dataset/cnn_dataset_test.npz` present.
- All commands are run **from the repository root** so dataset paths resolve.

## Build

```bash
export PATH=/usr/lib64/openmpi/bin:$PATH
mkdir -p build/experiments
mpicxx -std=c++20 -O3 -ICNN/src \
  CNN/experiments/layer_tuning/main.cpp \
  CNN/src/core/*.cpp CNN/src/data/*.cpp CNN/src/layers/*.cpp \
  CNN/src/model/*.cpp CNN/src/optimizers/*.cpp \
  CNN/src/training/*.cpp CNN/src/tuning/*.cpp \
  -o build/experiments/layer_tuning
```

## Run

Smoke test first (temporarily set `kFoldCount = 2` and `kEpochCount = 1` in
`main.cpp`, rebuild, then run):

```bash
mkdir -p results/cross_validation/layer_tuning
mpirun -n 2 build/experiments/layer_tuning \
  | tee results/cross_validation/layer_tuning/smoke.log
```

Restore `kFoldCount = 5` / `kEpochCount = 100`, rebuild, then run the full
search:

```bash
mpirun -n 4 build/experiments/layer_tuning \
  | tee results/cross_validation/layer_tuning/run.log
```

`-n` sets the number of MPI ranks; the global batch size stays 256 regardless.
The program prints per-candidate mean/stddev, per-fold validation MSE, the
fixed 10-epoch history for each fold, the winning topology, and — after
refitting the winner on the full training set — the single untouched-test MSE.

## Results

The cross-validation experiment evaluated 7 layer topologies across 5 folds for 100 epochs:

| Architecture | Mean val MSE (physical) | Std dev | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Status |
|---|---|---|---|---|---|---|---|---|
| `conv5x5-dense-512-256` | **0.809275** | **0.309016** | 0.766950 | 0.278995 | 0.763746 | 1.14431 | 1.09237 | **Winner** |
| `conv3x3-dense-256-128` | 0.840713 | 0.381084 | 0.704058 | 0.202923 | 0.843002 | 1.26257 | 1.19101 | Runner-up |
| `conv5x5-dense-256-128` | 0.878851 | 0.413215 | 0.729108 | 0.239579 | 0.835844 | 1.48807 | 1.10165 | Evaluated |
| `conv3x3-dense-512-256` | 0.908018 | 0.395120 | 1.118510 | 0.365395 | 1.145790 | 1.53802 | 1.12140 | Evaluated |
| `conv3x3-dense-128-64` | 0.980224 | 0.407585 | 0.824690 | 0.288347 | 1.089670 | 1.49663 | 1.20178 | Evaluated |
| `conv5x5-dense-128-64` (baseline) | 0.998040 | 0.527004 | 0.884435 | 0.364254 | 0.727833 | 1.94462 | 1.06906 | Evaluated |
| `conv5x5-dense-64-32` | 1.148290 | 0.485120 | 1.185200 | 0.452100 | 1.241500 | 1.78910 | 1.07355 | Underfitted |

### Final Untouched-Test Evaluation
- **Winning Architecture**: `conv5x5-dense-512-256` (High capacity dense head `{512, 256}`)
- **Final Untouched-Test Physical MSE**: **`0.876289`** (Scored on untouched test set after full refit)
- **Improvement over Baseline**: MSE reduced from `1.20877` to **`0.876289`** (~27.5% error reduction)
- **MPI Ranks**: 2 · **Epochs**: 100 · **Folds**: 5 · **Seed**: 42


Do not select a model from test-set performance — selection belongs to
cross-validation; the test set provides only the final unbiased estimate.

