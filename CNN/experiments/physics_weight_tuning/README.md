# Physics-Weight (SIMM λ) Tuning Experiment

## Question

How strongly should the physical prior be enforced? The SIMM loss is

```
L = MSE(pred, target) + λ · mask(|α| ≤ 10°) · MSE(pred, standardized 2πα)
```

λ = 0 is the pure data-driven model, so this sweep also answers whether the
physics term helps at all. λ = 0.25 is the current production default.

## Search Space

A single axis, `physics-weight`, with seven candidates sharing every other
hyperparameter (architecture, LeakyReLU α = 0.05, Adam 1e-5, batch 64,
5 folds, seed 42):

| λ | Role |
|---|---|
| 0 | Pure data-driven **(paired-analysis reference)** |
| 0.05 | Weak prior |
| 0.1 | |
| 0.25 | **Production default** |
| 0.5 | |
| 1.0 | Strong prior |
| 2.0 | Deliberately too strong: should visibly distort the fit if λ matters |

Total runs = 7 candidates × 5 folds = **35 fold evaluations**.

## Method

Selection metric is the physical-unit validation MSE, but the conclusion
comes from the **paired difference against λ = 0** (same folds, same seeds),
with the same pre-registered rule as the activation and regularization
experiments: a candidate beats the reference only if it improves in **≥ 4/5
folds** and |mean paired diff| > its std.

The recorder additionally splits the validation MSE into the small-angle
population (|α| ≤ 10°, where the prior is active) and the past-stall
population (|α| > 10°): a real physics effect must concentrate inside the
mask, so this split separates signal from fold noise.

## Build & Run

From the repository root:

```bash
make cnn      # builds build/CNN/experiments/physics_weight_tuning
mkdir -p results/cross_validation/physics_weight_tuning
mpirun -n 4 ./build/CNN/experiments/physics_weight_tuning sweep 100
python3 CNN/experiments/physics_weight_tuning/analyze.py \
    results/cross_validation/physics_weight_tuning/sweep_physics.csv
```

Per-epoch gradient, weight-update, and activation statistics (for stability
analysis) are recorded **by default** under
`results/physics_weight_tuning/sweep/candidate_NNN/fold_NNN/`; disable with
`--no-diagnostics` as the last argument. Plots:

```bash
python3 CNN/analysis/plot_training_diagnostics.py \
    --input results/physics_weight_tuning/sweep \
    --output-dir results/physics_weight_tuning/sweep/plots
```

> ⚠️ **Dataset prerequisite.** Results are only meaningful on the dataset
> regenerated with `python3 build_dataset.py --seed 42` from the
> data-pipeline portability fix (see `dataset/README.md` on that branch).

## Output

`sweep_physics.csv`, one row per (candidate, fold):

| column | meaning |
|---|---|
| `physics_weight` | λ of the candidate |
| `train_mse` / `val_mse` | physical-unit MSE on the training / validation fold |
| `baseline_mse` | mean-predictor MSE on the validation fold |
| `val_mse_small_angle`, `small_angle_count` | validation MSE and count inside the physics mask (\|α\| ≤ 10°) |
| `val_mse_large_angle`, `large_angle_count` | validation MSE and count past the mask |

## Results & Analysis

The selection criterion is the lowest mean physical-unit validation MSE across
the five folds. The paired difference against lambda 0 is reported as a
diagnostic. Stability was checked over all five folds and all 100 epochs using
the committed diagnostics:

- `peak grad` is the largest `maximum_norm` in `gradient_norms.csv` for the
  `all` parameter scope.
- `peak weight norm` is the largest `mean_pre_update_norm` in
  `parameter_update_ratios.csv` for the `weights` scope.
- `peak post-act var` is the largest `variance` in
  `activation_statistics.csv` for the `post_activation` phase.
- Every value used in these checks was finite. A candidate was considered
  numerically stable when these quantities stayed finite and bounded, with no
  late-epoch runaway increase.

| lambda | mean CV val MSE | fold std | paired delta vs 0 | mean train MSE | val-train gap | peak grad | peak weight norm | peak post-act var |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.005601568 | 0.001228485 | 0.0000000 | 0.005183447 | +0.000418122 | 56.8711 | 15.8807 | 0.196683 |
| 0.05 | 0.005638606 | 0.001270011 | +0.0000370 | 0.005217444 | +0.000421162 | 59.1692 | 15.8804 | 0.198590 |
| 0.1 | 0.005630026 | 0.001247489 | +0.0000285 | 0.005214046 | +0.000415979 | 61.4693 | 15.8801 | 0.199118 |
| 0.25 | 0.005707657 | 0.001336724 | +0.0001061 | 0.005307874 | +0.000399784 | 68.3797 | 15.8791 | 0.199148 |
| 0.5 | 0.006026890 | 0.001334059 | +0.0004253 | 0.005637086 | +0.000389804 | 79.9217 | 15.8782 | 0.201724 |
| 1.0 | 0.006582661 | 0.001509834 | +0.0009811 | 0.006226180 | +0.000356481 | 103.0610 | 15.8777 | 0.204872 |
| 2.0 | 0.007408295 | 0.001698277 | +0.0018067 | 0.007103552 | +0.000304743 | 149.4420 | 15.8769 | 0.203259 |

### Selected Configuration

`lambda = 0.0` is selected because it has the lowest mean cross-validated
validation MSE, `0.005601568`. All candidates passed the finite-value
stability check, but increasing lambda increases the peak gradient norm from
`56.8711` at lambda 0 to `149.4420` at lambda 2.0 without improving validation
MSE. The selected run is not numerically explosive: its peak gradient norm is
`56.8711` at the beginning of training and the epoch-100 maximum is `1.6971`;
the peak weight norm is `15.8807`, compared with `15.8705` at epoch 1; and the
peak post-activation variance is `0.196683`. The weight norm changes by less
than `0.1%`, while the gradient norm decreases strongly rather than growing
late in training.

For the selected lambda, the validation MSE is also stable when split by the
physics mask: the count-weighted small-angle MSE is `0.004096627` over `1442`
samples, and the count-weighted large-angle MSE is `0.013603506` over `271`
samples. The corresponding lambda 2.0 values are `0.005843330` and
`0.015727326`, so the stronger prior worsens both populations rather than
providing a targeted improvement.

The top three non-zero values by paired difference are `0.1` (`+0.0000285`),
`0.05` (`+0.0000370`), and `0.25` (`+0.0001061`). None improves on the
lambda-0 reference, and the production default `lambda = 0.25` is not
supported by this sweep's lowest-MSE criterion.

### Runner-up Stability

The second-lowest-MSE configuration is `lambda = 0.05`, with mean
cross-validated validation MSE `0.005638606`. Its diagnostics are also finite
and non-explosive: peak gradient norm `59.1692` at epoch 1 versus `1.89056` at
epoch 100, peak weight norm `15.8804` versus `15.8706` at epoch 1, and peak
post-activation variance `0.198590`. The weight norm increase is approximately
`0.062%`, and the gradient norm decreases strongly during training. It passes
the same numerical stability checks as lambda 0.0 but has a higher validation
MSE.

### Binary Angle-of-Attack OLS

The committed tuning output contains two angle groups rather than per-angle
predictions. Following the requested binary analysis, define

```text
x = 0  for |AoA| <= 10 degrees
x = 1  for |AoA| > 10 degrees
MSE = m*x + q
```

The values below are count-weighted across the five validation folds. The
counts are `1442` for the low-angle group and `271` for the high-angle group
for every lambda. Since there are exactly two points in each regression,
`R^2 = 1.0` is algebraically guaranteed; it measures the low/high-angle
contrast and must not be interpreted as evidence of a continuous linear
dependence on angle.

| lambda | MSE, \|AoA\| <= 10 | MSE, \|AoA\| > 10 | m | q | R^2 |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.004096627 | 0.013603506 | 0.009506879 | 0.004096627 | 1.000 |
| 0.05 | 0.004107911 | 0.013777423 | 0.009669513 | 0.004107911 | 1.000 |
| 0.10 | 0.004105860 | 0.013734311 | 0.009628451 | 0.004105860 | 1.000 |
| 0.25 | 0.004180479 | 0.013827530 | 0.009647051 | 0.004180479 | 1.000 |
| 0.50 | 0.004469502 | 0.014307660 | 0.009838158 | 0.004469502 | 1.000 |

The low-angle intercept `q` is lowest at lambda 0.0, and the high-angle MSE
is also lowest at lambda 0.0. Increasing lambda therefore does not improve
either requested angle group in these results. The slope `m` remains positive
for every lambda, meaning the high-angle group has larger error, with the
largest contrast at lambda 0.50.

## Caveats

- The random split shares geometries between train and validation
  (see the dataset README), so absolute MSE values are optimistic; the
  *relative* paired comparison remains valid because all candidates share
  the same split.
- The λ = 2.0 extreme exists to prove the sweep spans the active region: if
  even λ = 2.0 does not move the metrics, the physics term is inert in this
  regime and λ* = 0 should be read as "the prior is redundant with the data",
  not as "physics is wrong".
