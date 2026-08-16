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

*To be filled after the sweep run (see the group synthesis meeting: report
the top-3 rates by paired difference).*

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
