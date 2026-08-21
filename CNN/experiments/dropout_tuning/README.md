# Dropout-Rate Tuning Experiment

## Question

Does inverted dropout in the dense head improve generalization, and at which
rate? The regularization experiment measured a train/validation gap of
+0.000413 against a 0.001397 fold spread — little overfitting for a penalty
to remove — so the pre-registered expectation is that rate* = 0; this sweep
tests whether a *stochastic* regularizer reaches the same verdict.

## Search Space

A single axis, `dropout-rate`, applied after each hidden dense activation
(`Dense(128) → act → Dropout` and `Dense(64) → act → Dropout`). Every other
hyperparameter is fixed (LeakyReLU α = 0.05, Adam 1e-5, SIMM λ = 0.25,
batch 64, 5 folds, seed 42):

| rate | Role |
|---|---|
| 0 | Exact baseline **(paired-analysis reference)** |
| 0.1 | Light masking |
| 0.2 | |
| 0.3 | |
| 0.5 | Classic strong rate: must visibly move the metrics if dropout matters |

Total runs = 5 candidates × 5 folds = **25 fold evaluations**.

The dropout recipe is present in **every** candidate, including rate = 0
(identity at runtime): a recipe consumes one layer seed during construction,
so a shared structure means every candidate starts from the same initial
weights and the paired differences measure only the masking.

## Implementation notes

- Inverted dropout: surviving activations are scaled by 1/(1−rate) during
  training, inference needs no rescaling and `predict()` always runs the
  identity path.
- Masks are a stateless hash of (run seed, epoch, global batch, global sample
  row, feature), so training results are **independent of the MPI rank
  count** — verified by a dedicated unit test and a 1-vs-2-rank training run.

## Build & Run

From the repository root:

```bash
make cnn      # builds build/CNN/experiments/dropout_tuning
mkdir -p results/cross_validation/dropout_tuning
mpirun -n 4 ./build/CNN/experiments/dropout_tuning sweep 100
python3 CNN/experiments/dropout_tuning/analyze.py \
    results/cross_validation/dropout_tuning/sweep_dropout.csv
```

Per-epoch gradient, weight-update, and activation statistics (for stability
analysis) are recorded **by default** under
`results/dropout_tuning/sweep/candidate_NNN/fold_NNN/`; disable with
`--no-diagnostics` as the last argument. Plots:

```bash
python3 CNN/analysis/plot_training_diagnostics.py \
    --input results/dropout_tuning/sweep \
    --output-dir results/dropout_tuning/sweep/plots
```

> ⚠️ **Dataset prerequisite.** Results are only meaningful on the dataset
> regenerated with `python3 build_dataset.py --seed 42` from the
> data-pipeline portability fix (see `dataset/README.md` on that branch).

## Output

`sweep_dropout.csv`, one row per (candidate, fold):

| column | meaning |
|---|---|
| `dropout_rate` | rate of the candidate |
| `train_mse` / `val_mse` | physical-unit MSE on the training / validation fold |
| `baseline_mse` | mean-predictor MSE on the validation fold |

The analysis script applies the shared decision rule (≥ 4/5 folds improved
and |mean paired diff| > its std) and reports the train/validation gap per
rate, which is the quantity dropout is supposed to shrink.

## Results & Analysis

The selection criterion here is the lowest mean physical-unit validation MSE
across the five folds. The paired difference against rate 0 is reported as a
diagnostic, not as a replacement for the requested cross-validated MSE
criterion. Stability was checked over all five folds and all 100 epochs using
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

| rate | mean CV val MSE | fold std | paired delta vs 0 | mean train MSE | val-train gap | peak grad | peak weight norm | peak post-act var |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.005834597 | 0.001125249 | 0.0000000 | 0.005421198 | +0.000413398 | 37.8413 | 15.8782 | 0.186070 |
| 0.1 | 0.006686550 | 0.001184206 | +0.0008520 | 0.006390211 | +0.000296338 | 39.0003 | 15.8840 | 0.157550 |
| 0.2 | 0.007630465 | 0.000961879 | +0.0017959 | 0.007235922 | +0.000394543 | 38.4087 | 15.8842 | 0.153652 |
| 0.3 | 0.008466757 | 0.001275233 | +0.0026322 | 0.008097506 | +0.000369251 | 44.6056 | 15.8879 | 0.151285 |
| 0.5 | 0.011764955 | 0.002912000 | +0.0059304 | 0.011353900 | +0.000411055 | 47.4447 | 15.8899 | 0.185163 |

### Selected Configuration

`dropout rate = 0.0` is selected because it has the lowest mean
cross-validated validation MSE, `0.005834597`. All candidates passed the
finite-value stability check. The selected run is not numerically explosive:
its peak gradient norm is `37.8413` at the beginning of training and the
epoch-100 maximum is `2.69201`; the peak weight norm is `15.8782`, compared
with `15.8705` at epoch 1; and the peak post-activation variance is `0.186070`.
The weight norm changes by less than `0.1%`, and the gradient norm decreases
instead of growing late in training. The positive dropout rates increase the
validation MSE, so the lowest-MSE choice is also the least noisy choice for
this fixed epoch budget.

The top three non-zero rates by paired difference are `0.1` (`+0.0008520`),
`0.2` (`+0.0017959`), and `0.3` (`+0.0026322`). None improves on the rate-0
reference.

### Runner-up Stability

The second-lowest-MSE configuration is `dropout rate = 0.1`, with mean
cross-validated validation MSE `0.006686550`. Its diagnostics are also finite
and non-explosive: peak gradient norm `39.0003` at epoch 1 versus `2.26434` at
epoch 100, peak weight norm `15.8840` versus `15.8704` at epoch 1, and peak
post-activation variance `0.157550`. The weight norm increase is approximately
`0.086%`, and the gradient norm decreases strongly during training. Thus the
runner-up does not fail the numerical stability checks; it is rejected only
because its validation MSE is higher than the selected rate 0.0.

## Caveats

- The random split shares geometries between train and validation (see the
  dataset README), so absolute MSE values are optimistic; the *relative*
  paired comparison remains valid because all candidates share the split.
- Absolute fold values are **not** comparable with the other experiments'
  baselines: the extra dropout recipes shift the per-recipe layer-seed
  sequence, so the dense layers start from a different initialization
  realization than the shared no-dropout blueprint (the regularization gap
  quoted above is therefore a qualitative reference, not an expected value).
  Only the paired differences within this sweep carry meaning.
- With ~1700 training samples and a small head, dropout mostly adds gradient
  noise at a fixed epoch budget; if no rate wins, the honest conclusion is
  "no overfitting to remove", mirroring the L1/L2 verdict.
