# Weight-Regularization (L1/L2) Tuning — Large Architecture (`conv5x5-dense-{1024,512,256,128}`)

## Question

Does L1 or L2 weight regularization improve the cross-validated validation MSE
of the `conv5x5-dense-1024-512-256-128` network trained at `lr = 1e-5`?

The earlier small-topology experiment answered "no" on the
smaller `{128,64}` topology (**930 k parameters**: conv 208, dense 921 728 +
8 256 + 65) at the same learning rate. This network is **~8.7× larger**
(**8.065 M parameters**: conv 208, dense 7 375 872 + 524 800 + 131 328 +
32 896 + 129; the first dense takes 7 202 inputs because a
`ConcatenateLayer` appends 2 scalar features to the 7 200 flattened
convolution outputs). The rationale for repeating the sweep here is that it
is not obvious whether capacity alone changes the regularization answer when
the learning rate is fixed. The honest prior expectation is still λ* = 0:
at `lr = 1e-5` the network barely moves from its initialization during the
100-epoch budget and overfitting is unlikely regardless of parameter count.
**The reference sweep confirms this expectation.** Every nonzero λ worsens validation
MSE monotonically on both axes. As a by-product, the λ = 0 reference here
(mean val MSE **0.004419**) is **23.5 % lower** than the earlier experiment's
reference (mean val MSE 0.005776), confirming directly
that the larger architecture generalises better on this task.

## Search Space

Two independent one-dimensional sweeps (L1 alone, L2 alone), each with λ = 0
as its reference. The C++ executable builds one `ParameterGrid` per axis and
evaluates both grids through `CrossValidator::tune()` in a single MPI process.
There are **16 candidate configurations** in total: 8 L1 choices and 8 L2
choices. The zero-regularization candidate is intentionally present in both
grids so each axis has an independent `CandidateResult` and diagnostics tree.

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
# Build and run from the repository root.
cmake -S CNN -B build/CNN -DCMAKE_BUILD_TYPE=Release
cmake --build build/CNN --target regularization_tuning --parallel

# Run both complete ParameterGrid sweeps in one C++ process.
mpirun -n 16 build/CNN/experiments/regularization_tuning \
    --epochs 100 --folds 5 --seed 42 \
    --train-path dataset/cnn_dataset_train.npz \
    --results-dir results/cross_validation/regularization_tuning \
    --diagnostics
```

For a short validation run, `--smoke` uses two folds, two epochs, validation
after every epoch, and the first two lambda choices on each axis. The normal
invocation above evaluates all eight choices on both axes.

Per-epoch diagnostics are written under
`<results-dir>/regularization_tuning/<axis>/candidate_NNN/fold_MMM/`
(seven files: `metadata.json`, `epoch_metrics.csv`, `gradient_norms.csv`,
`parameter_update_ratios.csv`, `learning_rate_steps.csv`,
`activation_statistics.csv`, `activation_histograms.csv`).

## Output

One aggregate CSV per axis, one row per candidate and fold:

- `sweep_l1.csv`
- `sweep_l2.csv`

| column | meaning |
|---|---|
| `candidate` | `ParameterGrid` candidate name |
| `l1_weight` / `l2_weight` | regularization coefficients |
| `fold` | zero-based fold index |
| `train_mse` / `val_mse` | physical-unit MSE on train / validation split |
| `baseline_mse` | mean-predictor MSE on the validation fold |
| `l1_penalty` / `l2_penalty` | regularization term value at end of training |
| `sum_w2` | Σw² (sum of squared weights at end of training) |
| `weight_change_norm` | ‖w_final − w_init‖ |
| `epochs` | epochs completed |

Aggregate CSVs: `sweep_l1.csv`, `sweep_l2.csv`.
Per-checkpoint histories are written to `training_history_l1.csv` and
`training_history_l2.csv`. To generate the committed plot set after a
run, use:

```bash
python3 CNN/experiments/regularization_tuning/plot_results.py \
    --results-dir results/cross_validation/regularization_tuning
```

## Reference Results

The tables below are retained from the CINECA Leonardo DCGP run on 1 node and
16 MPI ranks (OpenMPI 4.1.6, gcc 12.2.0), job 53877281. That run used the same
model, grids, and fixed hyperparameters. The current executable intentionally
evaluates the λ = 0 candidate independently on each axis, so its full-run wall
time should be measured separately. The recorded reference run took **12 h
08 min 49 s** and used **~194** core-hours (16 cores × 12.15 h).

Its historical per-candidate times were ~48–54 min (l2_1e-01 was the slowest at
53.6 min; the λ = 0 reference ran in 42.0 min).

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

The conclusion matches the earlier small-topology experiment on the
`{128,64}` topology: at `lr = 1e-5`, the network does not overfit in 100
epochs regardless of the number of parameters, and there is nothing for a
weight penalty to correct.

**Comparison with the earlier experiment.** The λ = 0 reference here (mean
val MSE 0.004419 ± 0.000794) is **23.5 % lower** than its λ = 0 reference
(mean val MSE 0.005776 ± 0.001147). This is a
direct, independent confirmation of the layer-tuning winner: the
`{1024,512,256,128}` architecture generalises better than `{128,64}` at the
same learning rate and fold plan, by a margin that exceeds both experiments'
fold standard deviations.

**Fold consistency.** The cross-fold standard deviation (0.000794) is **18 %**
of the mean (0.004419). This is substantially more consistent than the 45 %
figure measured in `physics_weight_tuning` at lr = 1e-3 (std 0.001228 on mean
0.005602): at lr = 1e-5, training is slower but more stable across folds.

**Cluster margin.** Job 53877281 used 12 h 08 min of the 13 h 00 min allocated
(93.5 %). Anyone relaunching this sweep should allocate at least 15 h, or
reduce the number of candidates per job.

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
- The `parameter_update_ratios.csv` diagnostics report `mean_ratio` =
  `mean_update_norm / mean_pre_update_norm` per layer per epoch, broken out
  by scope (`all`, `weights`, `biases`). When averaged across **all scopes
  and last 10 epochs** the λ = 0 figure is **≈1.9e-3**. This aggregate is
  dominated by the `biases` scope: bias pre-update norms are ≈1e-4–5e-3
  (small absolute values) while bias update-step magnitudes are similar to
  those of weights, so the per-bias ratio reaches ≈3–6e-3. The
  **weights-only** ratio at the same epochs is **≈2.9e-5**; at epoch 9 it
  is **≈1.5e-5**. A reference figure of **~3.9e-5** appears in an older
  comment in `main.cpp` (attributed to `physics_weight_tuning_lr1e3`'s
  README at lr = 1e-5); the diagnostics for that experiment are not in the
  current repository tree, so the exact scope and epoch used there cannot
  be verified. The weights-only figures here (1.5e-5 at epoch 9, 2.9e-5 at
  epoch 91–100) are in the same order of magnitude as that reference and are
  consistent with it coming from a weights-scope measurement. **The
  discrepancy between the 1.9e-3 aggregate and the ~3.9e-5 reference is
  fully explained by scope aggregation (biases inflate the mean), not by an
  architecture change or Adam dynamics difference.**

## Diagnostics Path

```
<results-dir>/regularization_tuning/
  l1/
    candidate_000/ … candidate_007/
      fold_000/ … fold_004/
  l2/
    candidate_000/ … candidate_007/
      fold_000/ … fold_004/
        metadata.json
        epoch_metrics.csv
        gradient_norms.csv
        parameter_update_ratios.csv
        learning_rate_steps.csv
        activation_statistics.csv
        activation_histograms.csv
```

The λ = 0 reference has `candidate_000` under both `l1/` and `l2/`, because
each `ParameterGrid` produces its own complete `SearchResult`.

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

- The earlier small-topology source and result CSVs were removed when this
  larger topology became the canonical `regularization_tuning` experiment.
