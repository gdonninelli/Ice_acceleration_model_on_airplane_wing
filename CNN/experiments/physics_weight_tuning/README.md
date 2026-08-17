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

*To be filled after the sweep run (see the group synthesis meeting: report
the top-3 λ values by paired difference).*

## Caveats

- The random split shares geometries between train and validation
  (see the dataset README), so absolute MSE values are optimistic; the
  *relative* paired comparison remains valid because all candidates share
  the same split.
- The λ = 2.0 extreme exists to prove the sweep spans the active region: if
  even λ = 2.0 does not move the metrics, the physics term is inert in this
  regime and λ* = 0 should be read as "the prior is redundant with the data",
  not as "physics is wrong".
