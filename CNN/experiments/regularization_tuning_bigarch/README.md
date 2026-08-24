# Weight-Regularization (L1/L2) Tuning — Large Architecture (`conv5x5-dense-{1024,512,256,128}`)

## Question

Does L1 or L2 weight regularization improve the cross-validated validation MSE
of the `conv5x5-dense-1024-512-256-128` network trained at `lr = 1e-5`?

The original `CNN/experiments/regularization_tuning` answered "no" on the
smaller `{128,64}` topology (~75 k parameters) at the same learning rate. The
rationale for repeating the sweep here is that this network is roughly **100×
larger** (8.06 M parameters), and it is not obvious whether capacity alone
changes the regularization answer when the learning rate is fixed. The
honest prior expectation going in is still λ* = 0: at `lr = 1e-5` the
weight-update ratio is small, so the network barely moves from its
initialization during the 100-epoch budget and overfitting is unlikely.
**The sweep confirms this expectation.** Every nonzero λ worsens
validation MSE monotonically on both axes.

## Search Space

Two independent one-dimensional sweeps (L1 alone, L2 alone), each with λ = 0
as the shared reference. The orchestrator runs λ = 0 once and reuses the CSV
for both axes, so **15 unique training runs** produce 16 result rows.

- **L1**: `{0, 6.75e-7, 2.13e-6, 6.75e-6, 2.13e-5, 6.75e-5, 2.13e-4, 6.75e-4}`
- **L2**: `{0, 1e-4, 3.16e-4, 1e-3, 3.16e-3, 1e-2, 3.16e-2, 1e-1}`

All other hyperparameters are fixed: `conv5x5-dense-1024-512-256-128`,
LeakyReLU α = 0.05, Adam lr = 1e-5, physics weight = 0.25, batch 64,
5 folds, seed 42, 100 epochs.

## Method

Selection metric: mean cross-validated validation MSE (physical units).

**Paired-difference criterion** (pre-registered, identical to the original
experiment and to `physics_weight_tuning`): a candidate beats the reference
only when it improves in **≥ 4/5 folds** and the absolute mean paired
difference exceeds its own standard deviation. This guards against fold-noise
flukes.

## Build and Run

```bash
# Build (from repo root, out-of-source build in $WORK)
cmake -S . -B /path/to/build && cmake --build /path/to/build \
    --target regularization_tuning_bigarch -j$(nproc)

# Run one candidate
mpirun -n 16 /path/to/build/experiments/regularization_tuning_bigarch \
    --mode cv --axis l2 --lambda 1e-3 \
    --epochs 100 --folds 5 --seed 42 \
    --train-path dataset/cnn_dataset_train.npz \
    --results-dir results/sweep \
    --diagnostics

# Run the full sweep via the orchestrator (resumes automatically on timeout)
python3 CNN/experiments/regularization_tuning_bigarch/orchestrator.py \
    --binary /path/to/build/experiments/regularization_tuning_bigarch \
    --ranks 16 --epochs 100 --folds 5 --seed 42 \
    --diagnostics --results-dir results/sweep
```

Per-epoch diagnostics are written under
`<results-dir>/regularization_tuning_bigarch/<label>/candidate_000/fold_NNN/`
(six files: `metadata.json`, `epoch_metrics.csv`, `gradient_norms.csv`,
`parameter_update_ratios.csv`, `activation_statistics.csv`,
`activation_histograms.csv`).

## Output

One CSV per candidate (e.g. `l2_1e-03.csv`), one row per fold:

| column | meaning |
|---|---|
| `candidate` | label string (`l1_2e-05`, etc.) |
| `l1_weight` / `l2_weight` | regularization coefficients |
| `fold` | fold index (0–4) |
| `train_mse` / `val_mse` | physical-unit MSE on train / validation split |
| `baseline_mse` | mean-predictor MSE on the validation fold |
| `l1_penalty` / `l2_penalty` | regularization term value at end of training |
| `sum_w2` | Σw² (sum of squared weights at end of training) |
| `weight_change_norm` | ‖w_final − w_init‖ |
| `epochs` | epochs completed |

Aggregate CSVs: `sweep_l1_bigarch.csv`, `sweep_l2_bigarch.csv`.

## Results

Swept on CINECA Leonardo DCGP partition, 1 node, 16 MPI ranks (OpenMPI 4.1.6,
gcc 12.2.0), job 53877281. Total wall time: **12 h 08 min 49 s**; total
core-hours: **~194** (16 cores × 12.15 h). Average per candidate: ~48–54 min
(l2_1e-01 was the slowest at 53.6 min; the λ = 0 reference ran in 42.0 min
because Adam's first step overhead is front-loaded and subsequent candidates
benefit from warm caches on the same node).

**Reference (λ = 0): per-fold validation MSE**

| fold | val MSE | train MSE |
|---:|---:|---:|
| 0 | 0.0028714 | 0.0040525 |
| 1 | 0.0045082 | 0.0040377 |
| 2 | 0.0048860 | 0.0034491 |
| 3 | 0.0047752 | 0.0036525 |
| 4 | 0.0050525 | 0.0040140 |
| **mean** | **0.0044187** | **0.0038412** |
| std | 0.0007936 | 0.0002399 |

### L1 Axis

Paired difference Δ = mean(val_MSE_ref − val_MSE_λ): positive means λ improves on the reference.

| λ | mean val MSE | fold std | Δ vs λ=0 | Δ std | n folds improved | beats λ=0 | val−train gap | Σw² | ‖w−w₀‖ | peak grad | peak w norm | peak act var |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0044187 | 0.0007936 | 0.0000000 | 0.0000000 | — | — | +0.0005775 | 2992.6 | 0.91 | 22.54 | 42.36 | 0.1441 |
| 6.75e-7 | 0.0044240 | 0.0007880 | −0.0000053 | 0.0000688 | 2/5 | no | +0.0005677 | 2672.4 | 10.17 | 22.54 | 42.35 | 0.1441 |
| 2.13e-6 | 0.0044364 | 0.0008001 | −0.0000178 | 0.0000297 | 2/5 | no | +0.0005621 | 2408.7 | 15.38 | 22.54 | 42.34 | 0.1441 |
| 6.75e-6 | 0.0044775 | 0.0008110 | −0.0000588 | 0.0000317 | 0/5 | no | +0.0005694 | 2035.2 | 21.45 | 22.54 | 42.33 | 0.1441 |
| 2.13e-5 | 0.0045395 | 0.0008318 | −0.0001209 | 0.0000621 | 0/5 | no | +0.0005452 | 1592.8 | 27.87 | 22.54 | 42.32 | 0.1442 |
| 6.75e-5 | 0.0046843 | 0.0009019 | −0.0002656 | 0.0001521 | 0/5 | no | +0.0004556 | 1163.5 | 33.63 | 22.54 | 42.29 | 0.1447 |
| 2.13e-4 | 0.0050997 | 0.0009500 | −0.0006810 | 0.0002229 | 0/5 | no | +0.0002851 | 815.0 | 38.08 | 22.55 | 42.25 | 0.1492 |
| 6.75e-4 | 0.0064617 | 0.0011891 | −0.0020431 | 0.0005135 | 0/5 | no | +0.0001513 | 587.0 | 41.18 | 22.61 | 42.20 | 0.1654 |

### L2 Axis

| λ | mean val MSE | fold std | Δ vs λ=0 | Δ std | n folds improved | beats λ=0 | val−train gap | Σw² | ‖w−w₀‖ | peak grad | peak w norm | peak act var |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0044187 | 0.0007936 | 0.0000000 | 0.0000000 | — | — | +0.0005775 | 2992.6 | 0.91 | 22.54 | 42.36 | 0.1441 |
| 1e-4 | 0.0044831 | 0.0008186 | −0.0000644 | 0.0000618 | 1/5 | no | +0.0005715 | 2225.2 | 14.94 | 22.54 | 42.34 | 0.1441 |
| 3.16e-4 | 0.0045800 | 0.0008590 | −0.0001613 | 0.0000990 | 0/5 | no | +0.0005587 | 1808.1 | 20.30 | 22.54 | 42.33 | 0.1441 |
| 1e-3 | 0.0047676 | 0.0009316 | −0.0003489 | 0.0001778 | 0/5 | no | +0.0005171 | 1374.7 | 25.68 | 22.54 | 42.31 | 0.1442 |
| 3.16e-3 | 0.0050917 | 0.0009918 | −0.0006730 | 0.0002758 | 0/5 | no | +0.0003939 | 1008.7 | 30.14 | 22.54 | 42.27 | 0.1449 |
| 1e-2 | 0.0059023 | 0.0010995 | −0.0014836 | 0.0004261 | 0/5 | no | +0.0001915 | 748.4 | 33.48 | 22.55 | 42.23 | 0.1515 |
| 3.16e-2 | 0.0069936 | 0.0012170 | −0.0025749 | 0.0005769 | 0/5 | no | +0.0001567 | 621.1 | 35.56 | 22.70 | 42.18 | 0.1625 |
| 1e-1 | 0.0084531 | 0.0010800 | −0.0040345 | 0.0006053 | 0/5 | no | +0.0001140 | 582.3 | 36.46 | 24.08 | 42.14 | 0.1686 |

Diagnostic columns: `peak grad` = largest `maximum_norm` across all layers and
folds in `gradient_norms.csv`; `peak w norm` = largest `mean_pre_update_norm`
for the weights scope in `parameter_update_ratios.csv`; `peak act var` =
largest `variance` for the `post_activation` phase in
`activation_statistics.csv`. All values are finite; no numerical instability
was observed in any candidate.

### Selected Configuration

**λ* = 0** on both axes. Every nonzero λ degrades cross-validated validation
MSE monotonically. No candidate satisfies the pre-registered criterion (≥ 4/5
folds improved **and** |Δmean| > Δstd). The closest are L1 λ = 6.75e-7 and
λ = 2.13e-6, each improving in 2/5 folds with a mean difference of −5.3e-6
and −1.8e-5 respectively — both smaller in magnitude than their own standard
deviation, and below the 4/5-fold threshold.

The conclusion matches the original `regularization_tuning` experiment on the
`{128,64}` topology: at `lr = 1e-5`, the network does not overfit in 100
epochs regardless of the number of parameters, and there is nothing for a
weight penalty to correct.

**Regularization mechanics observed:**

- The val−train gap decreases with λ (from +5.8e-4 at λ = 0 to +1.1e-4 at
  L2 λ = 0.1), but training MSE rises faster than val MSE, indicating that
  the penalty impairs training more than it suppresses overfitting.
- Σw² drops by 5× from λ = 0 to the strongest candidates (2993 → 582),
  confirming active weight shrinkage. ‖w − w₀‖ increases in parallel (from
  0.91 to ~36–41), showing that the regularizer pulls weights away from their
  Xavier initialization rather than keeping them there.
- Peak gradient norms are nearly identical across all L1 candidates and most
  L2 candidates (≈22.54), only rising at L2 λ = 3.16e-2 (22.70) and
  λ = 0.1 (24.08). The gradient-clip threshold (1.0 per-layer) is never
  approached for any candidate other than at epoch 1 (initial forward pass
  scale), which is consistent with stable low-lr training.
- Mean weight-update ratios at the last 10 epochs are ≈1.9e-3 for λ = 0,
  decreasing to ≈5.3e-4 at the strongest penalties. The λ = 0 value is
  notably larger than the ~3.9e-5 figure from `physics_weight_tuning`'s
  lr = 1e-5 entry (recorded on the old `{128,64}` topology): the two
  architectures differ by ~100× in parameter count, and the ratio reflects
  different per-layer weight magnitudes under Xavier initialization at
  different fan-in values, not a change in the learning rate.

## Diagnostics Path

```
<results-dir>/regularization_tuning_bigarch/
  <label>/          # e.g. l1_2e-05, l2_1e-03
    candidate_000/
      fold_000/ … fold_004/
        metadata.json
        epoch_metrics.csv
        gradient_norms.csv
        parameter_update_ratios.csv
        activation_statistics.csv
        activation_histograms.csv
```

For the λ = 0 reference, diagnostics are stored under `l1_0/` only. The
`l2_0.csv` file is a copy of `l1_0.csv` (same training run, reused by the
orchestrator to avoid a redundant identical computation); there is no
separate `l2_0/` diagnostics directory.

## Caveats

- **No held-out test set.** Both `training_dataset_path` and
  `validation_dataset_path` in the metadata point to the same file
  (`cnn_dataset_train.npz`). The 5-fold cross-validation splits that file
  in memory: at each fold, ~80 % of the samples serve as the training split
  and ~20 % as the validation split. Every sample appears in a validation
  split exactly once across the five folds. Because the same geometries
  appear in both roles (across folds) and there is no separate held-out test
  file, the absolute MSE values are optimistic; the **relative** paired
  comparisons across candidates remain valid because all candidates use the
  same splits and seeds.

- **Fixed epoch budget.** 100 epochs is the same budget as the original
  experiment and is sufficient to measure regularization effects in this
  regime. No early-stopping or learning-rate schedule was used; the result
  holds for Adam with fixed `lr = 1e-5` at exactly 100 epochs.

- **Grid coverage.** The L1 grid ends at 6.75e-4 (mean val MSE 0.0065)
  and the L2 grid at 0.1 (mean val MSE 0.0085). Both axes show clear
  monotone degradation well before the grid boundary, so the absence of a
  benefit is not an artifact of insufficient coverage.

- The original `CNN/experiments/regularization_tuning` (Giulio Donninelli,
  on `main`) is untouched. This experiment lives in a separate directory and
  does not modify any file in that directory.
