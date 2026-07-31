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
- Dense head hidden widths (`{64, 32}` up to `{2048, 1024}`),
- Dense head depth (2, 3, or 4 hidden layers).

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
| `conv3x3-dense-256-128` | `make_blueprint(8, 3, {256, 128})` | wider head + `3x3` |
| `conv5x5-dense-512-256` | `make_blueprint(8, 5, {512, 256})` | widest dense head |
| `conv3x3-dense-512-256` | `make_blueprint(8, 3, {512, 256})` | widest head + `3x3` |
| `conv5x5-dense-64-32` | `make_blueprint(8, 5, {64, 32})` | narrower dense head |
| `conv5x5-dense-512-256-128` | `make_blueprint(8, 5, {512, 256, 128})` | +1 hidden layer (depth 3) |
| `conv5x5-dense-512-256-128-64` | `make_blueprint(8, 5, {512, 256, 128, 64})` | +2 hidden layers (depth 4) |
| `conv5x5-dense-768-384` | `make_blueprint(8, 5, {768, 384})` | wider than 512-256 |
| `conv5x5-dense-1024-512` | `make_blueprint(8, 5, {1024, 512})` | wider than 512-256 |
| `conv5x5-dense-1536-768` | `make_blueprint(8, 5, {1536, 768})` | wider than 512-256 |
| `conv5x5-dense-2048-1024` | `make_blueprint(8, 5, {2048, 1024})` | widest head tested |

Total runs = 13 candidates × 5 folds = **65 fold evaluations**, plus one final
refit + test evaluation for the winner.

> **Capacity note.** The dataset has only 40 training samples (32 per fold),
> while the `{512, 256}` head already holds ~3.8 M parameters and `{2048, 1024}`
> holds ~15 M. Depth candidates add very few parameters (the first `dense`
> dominates), while width candidates scale parameters roughly linearly. The gaps
> between candidates are of the same order as the standard deviations, so treat
> the results as exploratory. The next step to make them more robust is
> **dropout** (a separate experiment).

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

The cross-validation experiment evaluated 13 layer topologies across 5 folds for
100 epochs (`mpirun -n 4`, seed 42):

| Architecture | Mean val MSE (physical) | Std dev | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Status |
|---|---|---|---|---|---|---|---|---|
| **`conv5x5-dense-1024-512`** | **0.800921** | **0.317712** | 0.752245 | 0.312365 | 0.998310 | 1.25975 | 0.681928 | 🥇 **1st (Winner)** |
| **`conv5x5-dense-512-256`** | **0.809292** | **0.308987** | 0.766950 | 0.279081 | 0.763746 | 1.14431 | 1.09237 | 🥈 **2nd** |
| **`conv3x3-dense-256-128`** | **0.840713** | **0.381084** | 0.704058 | 0.202923 | 0.843003 | 1.26257 | 1.19101 | 🥉 **3rd** |
| `conv5x5-dense-256-128` | 0.878851 | 0.413215 | 0.729108 | 0.239579 | 0.835844 | 1.48807 | 1.10165 | Evaluated |
| `conv5x5-dense-512-256-128-64` | 0.927782 | 0.329733 | 1.06036 | 0.314560 | 0.958404 | 1.30740 | 0.998188 | Depth 4 |
| `conv5x5-dense-512-256-128` | 0.941706 | 0.461663 | 1.24334 | 0.222712 | 0.878465 | 1.59325 | 0.770761 | Depth 3 |
| `conv5x5-dense-768-384` | 0.968328 | 0.395494 | 0.856030 | 0.370653 | 0.919406 | 1.59465 | 1.10090 | Wider |
| `conv5x5-dense-1536-768` | 0.971953 | 0.381773 | 0.816253 | 0.338724 | 1.04058 | 1.47247 | 1.19173 | Wider |
| `conv3x3-dense-128-64` | 0.980224 | 0.407585 | 0.824690 | 0.288347 | 1.08967 | 1.49663 | 1.20178 | Evaluated |
| `conv5x5-dense-64-32` | 0.993322 | 0.473087 | 1.61598 | 0.175147 | 0.965751 | 1.23959 | 0.970145 | Evaluated |
| `conv5x5-dense-128-64` (baseline) | 0.998040 | 0.527004 | 0.884435 | 0.364254 | 0.727833 | 1.94462 | 1.06906 | Baseline |
| `conv5x5-dense-2048-1024` | 1.04082 | 0.437162 | 1.04850 | 0.272840 | 1.01536 | 1.60154 | 1.26586 | Widest |
| `conv3x3-dense-512-256` | 1.11857 | 0.409376 | 1.11851 | 0.365395 | 1.14579 | 1.53802 | 1.42515 | Evaluated |

### Depth vs width

**Depth did not help.** The deeper heads (`-512-256-128`, `-512-256-128-64`)
landed mid-pack (`0.942`, `0.928`), above the best 2-layer heads.

**Width does not scale monotonically.** Sweeping the conv5x5 2-layer head width
gives a jagged, non-monotonic curve rather than steady improvement:

| Head width | `64-32` | `128-64` | `256-128` | `512-256` | `768-384` | `1024-512` | `1536-768` | `2048-1024` |
|---|---|---|---|---|---|---|---|---|
| Mean val MSE | 0.993 | 0.998 | 0.879 | 0.809 | 0.968 | **0.801** | 0.972 | 1.041 |

`1024-512` is nominally best, but only `0.008` below `512-256` — about 2.5% of
its own standard deviation (`0.318`), while neighbours `768-384` and `1536-768`
are clearly *worse*. There is no reliable benefit to going wider than
`512-256`: the differences are dominated by fold noise on 40 samples, and the
largest head (`2048-1024`) is among the worst. Separating real signal from
noise here needs regularization (**dropout**) and/or more data, not more width.

### Final Untouched-Test Evaluation

- **Winning Architecture**: `conv5x5-dense-1024-512` (dense head `{1024, 512}`)
- **Final Untouched-Test Physical MSE**: **`0.723271`** (scored once on the
  untouched test set after refitting on the full training NPZ)
- **Improvement over baseline (CV mean)**: `0.998040` → **`0.800921`**
  (~19.8% reduction), but within one standard deviation
- **Caveat**: the winner beats `512-256` by only `0.008` in CV — well inside the
  noise — so this selection should be read as "≈ tied with 512-256", not as
  evidence that `1024-512` is genuinely better.
- **MPI Ranks**: 4 · **Epochs**: 100 · **Folds**: 5 · **Seed**: 42


Do not select a model from test-set performance — selection belongs to
cross-validation; the test set provides only the final unbiased estimate.

