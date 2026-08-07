# Layer-Architecture Tuning Experiment

> ⚠️ **Risultati non disponibili.** Questo esperimento è stato eseguito originariamente su
> un dataset che si è rivelato invalido: i file `.npz` committati nel repository
> contenevano rumore gaussiano anziché le Signed Distance Function dei profili alari
> (autocorrelazione lag-1 ≈ 0.0001 contro ≈ 0.999 attesa; nessuno dei target presente nei
> summary CFD). Tutte le misure e le conclusioni prodotte in quella sede sono state rimosse.
> L'infrastruttura dell'esperimento è valida e riutilizzabile. Le misure verranno rifatte
> sul dataset rigenerato (2142 campioni, 5 profili, verificato fisicamente: pendenza
> C_l vs α = 0.111/grado contro 0.1097 teorico).
> 
> In particolare, la conclusione precedente secondo cui la profondità della rete non
> portava benefici è **contraddetta** dalle misure preliminari sui dati rigenerati.

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
