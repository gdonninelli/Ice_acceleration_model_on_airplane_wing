# Layer-Architecture Tuning Experiment

> ⚠️ **Prerequisito sui dati.** I risultati riportati sono stati ottenuti sul dataset
> rigenerato con `python3 build_dataset.py --seed 42`. Quella correzione della pipeline
> è oggetto di una pull request separata e non è ancora su `main`.

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
- Dense head depth (2 to 5 hidden layers).

Selection uses sample-weighted **physical-unit validation MSE** minimized by
`CrossValidator`. The held-out test NPZ is never read during selection; it is
used exactly once to produce an unbiased estimate for the winning topology.

## Feature Trunk / Head

Each candidate is produced by
`make_blueprint(out_channels, kernel_size, hidden_widths)`:

- **Feature trunk:** `conv2d(out_channels, kernel_size, kernel_size)` →
  `leakyrelu` → `flatten`. The third `conv2d` argument is the stride, matching
  the documented experiment template, so kernel size and stride move together.
  With `5x5` on a 150×150 SDF this yields 8×30×30 = 7200 flattened features.
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
| Global batch size | 64 (MPI-global, does not scale with ranks) |
| Seed | 42 |
| Selection metric | physical-unit validation MSE |
| Training NPZ | `dataset/cnn_dataset_train.npz` (1713 samples) |
| Test NPZ (once) | `dataset/cnn_dataset_test.npz` (429 samples) |

Because the batch size is MPI-global, the number of ranks changes only the
wall-clock time, not the numbers.

## Candidates (`layer-architecture` axis)

| Label | Blueprint | Varies from baseline |
|---|---|---|
| `conv5x5-dense-128-64` | `make_blueprint(8, 5, {128, 64})` | baseline |
| `conv3x3-dense-128-64` | `make_blueprint(8, 3, {128, 64})` | kernel `5x5` → `3x3` |
| `conv5x5-dense-256-128` | `make_blueprint(8, 5, {256, 128})` | wider dense head |
| `conv3x3-dense-256-128` | `make_blueprint(8, 3, {256, 128})` | wider head + `3x3` |
| `conv5x5-dense-512-256` | `make_blueprint(8, 5, {512, 256})` | wider dense head |
| `conv3x3-dense-512-256` | `make_blueprint(8, 3, {512, 256})` | wider head + `3x3` |
| `conv5x5-dense-64-32` | `make_blueprint(8, 5, {64, 32})` | narrower dense head |
| `conv5x5-dense-512-256-128` | `make_blueprint(8, 5, {512, 256, 128})` | depth 3 |
| `conv5x5-dense-512-256-128-64` | `make_blueprint(8, 5, {512, 256, 128, 64})` | depth 4 |
| `conv5x5-dense-768-384` | `make_blueprint(8, 5, {768, 384})` | wider than 512-256 |
| `conv5x5-dense-1024-512` | `make_blueprint(8, 5, {1024, 512})` | wider than 768-384 |
| `conv5x5-dense-1536-768` | `make_blueprint(8, 5, {1536, 768})` | wider than 1024-512 |
| `conv5x5-dense-2048-1024` | `make_blueprint(8, 5, {2048, 1024})` | widest head tested |
| `conv5x5-dense-512-256-128-64-32` | `make_blueprint(8, 5, {512, 256, 128, 64, 32})` | depth 5 |
| `conv5x5-dense-1024-512-256-128` | `make_blueprint(8, 5, {1024, 512, 256, 128})` | wider, depth 4 |
| `conv5x5-dense-1024-512-256-128-64` | `make_blueprint(8, 5, {1024, 512, 256, 128, 64})` | wider, depth 5 |
| `conv5x5-dense-512-64` | `make_blueprint(8, 5, {512, 64})` | wide first layer, narrow second |
| `conv5x5-dense-256-128-64-32` | `make_blueprint(8, 5, {256, 128, 64, 32})` | narrow, depth 4 |

Total runs = 18 candidates × 5 folds = **90 fold evaluations**, plus one final
refit + test evaluation for the winner.

## Results

**Provenance.** The measurements below were produced by **Giulio Donninelli** on the
**CINECA Leonardo** cluster (SLURM job `51807287`), from this branch's
`CNN/experiments/layer_tuning/main.cpp`. The raw stdout is committed as
[`run_leonardo_51807287.log`](../../../results/cross_validation/layer_tuning/run_leonardo_51807287.log);
the extracted per-fold values are in
[`aggregated.csv`](../../../results/cross_validation/layer_tuning/aggregated.csv) and
[`summary.csv`](../../../results/cross_validation/layer_tuning/summary.csv).

Each fold trained on 1370–1371 samples and validated on 342–343, which matches the
regenerated dataset (1713 training samples). All 18 candidates were evaluated on the
**same fold plan**: `CrossValidator::tune` builds the folds once, before the candidate
loop, and reuses them for every candidate. The paired analysis below is therefore valid.

`Δ` is the per-fold difference against the `{128, 64}` baseline, averaged with the same
sample weighting; "folds better" counts how many of the 5 folds improved on the baseline.
`ratio` is `val_mse / baseline_mse`, where `baseline_mse` is the mean-predictor MSE
(predicting the training-fold mean). The layer-tuning binary on this branch does not emit
`baseline_mse`, so the per-fold values are taken from the activation-tuning experiment,
which uses the identical splitter (`RandomKFold(5, shuffle, seed 42)`) on the identical
dataset and therefore the identical folds; their sample-weighted mean is **0.4974**.

| candidate | val MSE | std | Δ vs `{128,64}` | std(Δ) | folds better | ratio |
|---|---|---|---|---|---|---|
| `conv5x5-dense-1024-512-256-128` | 0.004560 | 0.000946 | -0.001450 | 0.000318 | 5/5 | 0.0092 |
| `conv5x5-dense-1024-512-256-128-64` | 0.004626 | 0.000918 | -0.001384 | 0.000322 | 5/5 | 0.0093 |
| `conv5x5-dense-1024-512` | 0.004851 | 0.000911 | -0.001158 | 0.000251 | 5/5 | 0.0098 |
| `conv5x5-dense-512-256-128-64` | 0.004872 | 0.001125 | -0.001138 | 0.000426 | 5/5 | 0.0098 |
| `conv5x5-dense-512-256-128` | 0.004898 | 0.000970 | -0.001112 | 0.000346 | 5/5 | 0.0098 |
| `conv5x5-dense-768-384` | 0.004932 | 0.001099 | -0.001077 | 0.000340 | 5/5 | 0.0099 |
| `conv5x5-dense-512-256` | 0.005052 | 0.001067 | -0.000958 | 0.000326 | 5/5 | 0.0102 |
| `conv5x5-dense-512-256-128-64-32` | 0.005061 | 0.001031 | -0.000949 | 0.000377 | 5/5 | 0.0102 |
| `conv5x5-dense-512-64` | 0.005145 | 0.001014 | -0.000865 | 0.000229 | 5/5 | 0.0103 |
| `conv5x5-dense-1536-768` | 0.005307 | 0.001177 | -0.000703 | 0.001083 | 4/5 | 0.0107 |
| `conv5x5-dense-256-128` | 0.005447 | 0.001060 | -0.000563 | 0.000480 | 4/5 | 0.0110 |
| `conv5x5-dense-256-128-64-32` | 0.005517 | 0.001095 | -0.000493 | 0.000526 | 3/5 | 0.0111 |
| `conv5x5-dense-2048-1024` | 0.005673 | 0.001466 | -0.000337 | 0.001396 | 3/5 | 0.0114 |
| `conv5x5-dense-128-64` (baseline) | 0.006010 | 0.000823 | — | — | — | 0.0121 |
| `conv3x3-dense-256-128` | 0.006157 | 0.001051 | +0.000147 | 0.000565 | 2/5 | 0.0124 |
| `conv3x3-dense-128-64` | 0.006330 | 0.001099 | +0.000321 | 0.000312 | 1/5 | 0.0127 |
| `conv5x5-dense-64-32` | 0.006425 | 0.001292 | +0.000415 | 0.000529 | 2/5 | 0.0129 |
| `conv3x3-dense-512-256` | 0.006459 | 0.001318 | +0.000449 | 0.001281 | 2/5 | 0.0130 |

**Winner:** `conv5x5-dense-1024-512-256-128`, val MSE **0.004560 ± 0.000946**, better than
the baseline in **5 folds out of 5**, with `|Δ| = 0.001450` roughly 4.6× its own spread
across folds (0.000318) — the largest and most consistent margin in the grid. Refit on the
full training set, its untouched-test MSE is **0.002805**.

Every candidate beats the mean predictor by two orders of magnitude (ratio ≈ 0.009–0.013),
so all of them learn something; the grid separates good from better, not signal from noise.

### Is the winner interior to the grid, or on the boundary?

**Interior in depth, on the boundary in width.** These are different answers per axis and
the distinction matters:

- **Depth (interior).** In the `1024` family the grid brackets the winner on both sides:
  depth 2 (`{1024,512}`, 0.004851) and depth 5 (`{1024,512,256,128,64}`, 0.004626) are both
  worse than depth 4 (0.004560). The optimum in depth is genuinely enclosed.
- **Width (boundary).** `{1024, 512, 256, 128}` is the **widest depth-4 head tested**: no
  `{1536, …}` or `{2048, …}` deep candidate exists. Along that axis the winner sits on the
  edge of what was explored.

The boundary is, however, much less worrying than it was in the earlier probe, because the
depth-2 width sweep is complete and shows width **saturating and then reversing**:
0.006425 (`64-32`) → 0.006010 → 0.005447 → 0.005052 → 0.004932 (`768-384`) →
**0.004851** (`1024-512`, best) → 0.005307 (`1536-768`) → 0.005673 (`2048-1024`).
Width past 1024 hurts at depth 2. It is therefore unlikely — but **not proven** — that a
`{1536, 768, 384, 192}` head would beat the winner. Confirming it requires one more run.

For contrast: the previous probe's winner `{512, 256, 128, 64}` was the deepest *and*
widest candidate in that grid — a true boundary optimum. It is now interior, and beaten.

### Depth or capacity?

The grid contains both `{512,256}` and `{512,256,128,64}`, so the two effects separate.
**Both contribute, by comparable amounts, and they add up:**

| comparison | from → to | Δ | folds better |
|---|---|---|---|
| depth at width 512 | `{512,256}` → `{512,256,128,64}` | -0.000180 | 4/5 |
| depth at width 1024 | `{1024,512}` → `{1024,512,256,128}` | -0.000292 | 5/5 |
| width at depth 2 | `{512,256}` → `{1024,512}` | -0.000201 | 4/5 |
| width at depth 4 | `{512,256,128,64}` → `{1024,512,256,128}` | -0.000313 | 5/5 |

Going from `{512,256}` (0.005052) to the winner (0.004560) is worth -0.000492, which is
close to the sum of one depth step and one width step taken separately. Neither knob alone
explains the gain: **the answer is "both", not "depth" or "capacity"**. Note also that
depth pays off more at the wider setting (-0.000292 vs -0.000180), i.e. the two interact
mildly rather than being strictly independent.

Two boundary observations support the same picture:

- Depth stops paying at 5 layers: `{512,256,128,64,32}` (0.005061) is no better than
  `{512,256}` (0.005052), and `{1024,…,64}` is worse than `{1024,…,128}`.
- The first dense layer dominates: `{512,64}` (0.005145) is nearly as good as `{512,256}`
  (0.005052), so shrinking the *second* layer costs little.

### Kernel size

`5x5` beats `3x3` in all three paired comparisons, and the gap widens with head width:
-0.000321 at `128-64` (4/5 folds), -0.000710 at `256-128` (4/5), -0.001407 at `512-256`
(5/5). No `3x3` candidate beats the baseline.

## Caveats

> I risultati usano lo split train/test casuale (seed 42), in cui la stessa geometria
> (profilo, angolo) compare in entrambi i set con Reynolds diverso. Poiché il Reynolds non
> influenza il target in modo misurabile, questi campioni sono di fatto duplicati e le MSE
> di validazione sono ottimistiche. I confronti *relativi* fra candidati restano validi;
> i valori *assoluti* andranno rimisurati con uno split raggruppato.

- **Capacity.** `{512, 256}` holds ~3.8 M parameters and `{2048, 1024}` ~16.8 M, against
  1713 training samples. Depth candidates add few parameters (the first `dense` dominates);
  width candidates scale them roughly linearly. That the widest heads get *worse* is
  consistent with capacity, not depth, being the binding constraint at the top of the grid.
- **Spread.** Differences among the top candidates (0.00456–0.00505) are smaller than the
  fold-to-fold standard deviation (~0.001), which is why the paired per-fold comparison,
  not the raw means, carries the argument. The 5/5 sign consistency is the evidence.
- **No repeated seeds.** Each candidate was trained once per fold. The run does not
  separate architecture differences from training-seed noise.

## Requirements

- MPI toolchain (OpenMPI `mpicxx` / `mpirun`). On this machine:
  `export PATH=/usr/lib64/openmpi/bin:$PATH`.
- A C++20 compiler.
- `dataset/cnn_dataset_train.npz` and `dataset/cnn_dataset_test.npz` present, regenerated
  with `python3 build_dataset.py --seed 42` (see the note at the top).
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

Smoke test first (temporarily set `kFoldCount = 2` and `kEpochCount = 2` in `main.cpp`,
and cut the grid to a single candidate, rebuild, then run):

```bash
mkdir -p results/cross_validation/layer_tuning
mpirun -n 2 build/experiments/layer_tuning \
  | tee results/cross_validation/layer_tuning/smoke.log
```

Restore `kFoldCount = 5` / `kEpochCount = 100` and the full grid, rebuild, then run the
full search:

```bash
mpirun -n 4 build/experiments/layer_tuning \
  | tee results/cross_validation/layer_tuning/run.log
```

`-n` sets the number of MPI ranks; the global batch size stays 64 regardless, so the rank
count does not change the results.

### Reproducing the committed run (CINECA Leonardo)

The committed measurements come from SLURM job `51807287`, submitted by Giulio Donninelli.
The job's stdout was captured as `run_51807287.log` and is committed here as
`run_leonardo_51807287.log`. The rank count used on the cluster is not recorded in the log;
as noted above it does not affect the numbers, only the wall-clock time. On a SLURM cluster
the equivalent submission is:

```bash
# after building as above, from the repository root
srun --ntasks=<ranks> build/experiments/layer_tuning \
  | tee results/cross_validation/layer_tuning/run_leonardo_51807287.log
```

The program prints per-candidate mean/stddev, per-fold validation MSE, the fixed 10-epoch
history for each fold, the winning topology, and — after refitting the winner on the full
training set — the single untouched-test MSE.
