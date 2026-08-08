# Weight-Regularization Tuning Experiment

Cross-validation experiment that tunes the L1 (Lasso) and L2 (Ridge) penalties
added on `feature/l1-l2-regularization`, built on the typed `CrossValidator` /
`ParameterGrid` API documented in
[`cross_validation.md`](../../../cross_validation.md).

## Scientific Question

Holding architecture, optimizer, learning rate, physics weight and schedule
fixed, **does penalizing the weights improve generalization, and at which
lambda?**

The expected answer was stated before running: **probably lambda\* = 0**. The
train/validation gap measured on this dataset is `+0.000413` against a
`0.001397` spread across folds, positive in only 3 folds of 5 — there is no
overfitting for a penalty to remove. That is a result, not a failed experiment.

## Design

### Two one-dimensional sweeps, not a 2D grid

A Cartesian L1 x L2 grid would require 64 candidates, compared with 16 total
across the two one-dimensional sweeps, and would measure an interaction term
that cannot matter when neither main effect has room to act. Sweep A varies L2
with `l1_weight = 0`; sweep B varies L1 with `l2_weight = 0`. A 2D grid is
justified only if one of the two shows a real effect.

### Paired analysis, not a comparison of means

This is the point the experiment turns on. The fold-to-fold spread is ~0.0014,
larger than any effect expected here, so comparing `mean +/- std` between
candidates cannot conclude anything. Every candidate is evaluated on the **same
fold plan with the same seeds**, so the correct comparison is paired: for each
fold, `val_mse(lambda) - val_mse(0)`, then analyse the five differences.

Decision rule, fixed before looking at any result:

> A lambda beats 0 only if it improves in **at least 4 folds out of 5** and the
> mean paired difference is larger in magnitude than its own standard
> deviation. Otherwise lambda\* = 0.

### Fixed baseline

| Setting | Value |
|---|---|
| Architecture | `conv2d(8, 5, stride 5)` -> leakyrelu -> flatten -> dense 128 -> 64 -> 1 |
| Optimizer | Adam, learning rate `1e-5`, Adam's own `weight_decay` left at 0 |
| Loss | SIMM physics loss, weight `0.25` (itself a regularizer, so it is held fixed) |
| Folds | 5 (`RandomKFold`, shuffle, seed 42) |
| Epochs | 100 |
| Global batch size | 64 (~22 Adam steps per epoch on a 1370-sample fold) |
| Seed | 42 |
| Selection metric | paired difference of physical-unit validation MSE against lambda = 0 |
| Training NPZ | `dataset/cnn_dataset_train.npz` (1713 samples) |

The test NPZ is never read: this experiment compares candidates, and the
absolute number would come from a refit outside it.

## Epoch budget: 100, and why

Measured on the regenerated dataset, 8 MPI ranks: **1.16 s per fold-epoch**,
plus ~7.4 s to load the NPZ (the loader shells out to Python per array).
16 ranks was *slower* than 8 on this machine (oversubscription), so 8 is used
throughout.

A 500-epoch convergence probe on fold 0 with lambda = 0
([`convergence_fold0.csv`](../../../results/cross_validation/regularization_tuning/convergence_fold0.csv))
shows that **the model has not converged even at 500 epochs**:

| epoch | 10 | 50 | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|---|---|
| val MSE | 0.005094 | 0.003979 | 0.003675 | 0.003233 | 0.002696 | 0.003395 | 0.002413 |

Validation MSE is lower at epoch 500 than at epoch 100, despite noisy
intermediate increases. There is no sustained overfitting trend or clear
plateau to anchor the budget to in this single fold. 100 epochs was chosen
because:

- it is the budget every other experiment in this repository uses
  (`layer_tuning`, `activation_tuning`, `CNN/main.cpp` default), which keeps
  the numbers comparable;
- the paired design makes the lambda comparison valid at **any** fixed budget,
  since all candidates share it;
- at 100 epochs, the compute-only estimate is 8 candidates x 5 folds x 100
  epochs x 1.16 s = **~77 min per sweep**, before process and I/O overhead. At
  500 epochs the corresponding estimate is ~6.4 h per sweep, which is not
  proportionate to a question whose expected answer is zero.

**Caveat this creates**: if regularization only helps in a regime the model
reaches after 100 epochs, this experiment cannot see it. The 500-epoch probe
does not show a sustained overfitting onset, but it is a single noisy fold with
lambda = 0.

## Candidates

### Sweep A — L2 (Ridge), `l1_weight = 0`

Log-spaced at sqrt(10) over `[1e-4, 1e-1]`, plus 0 as reference. The band
brackets the measured balancing lambda2, the value at which the penalty
gradient `2*lambda2*|w|` equals the median data gradient: **0.0762 at
initialization, settling to 7e-4 - 1e-3** after epoch 10.

| label | lambda2 |
|---|---|
| `0` | 0 (reference) |
| `1e-04` | 1e-4 |
| `3e-04` | 3.16e-4 |
| `1e-03` | 1e-3 |
| `3e-03` | 3.16e-3 |
| `1e-02` | 1e-2 |
| `3e-02` | 3.16e-2 |
| `1e-01` | 1e-1 |

### Sweep B — L1 (Lasso), `l2_weight = 0`

**The L1 range is not the L2 range.** The L1 penalty gradient is
`lambda1*sign(w)`, independent of `|w|`, so the balancing value is the median
data gradient itself rather than `grad/(2|w|)`:

```
lambda1* = median|grad_data| = 2.13e-5   (measured at epoch 100 over all 930312 weights)
```

Centred there, with the same 3-decade width as sweep A, so the extremes sit at
1/30 and 30x the data gradient. Centring on the *late-training* gradient rather
than the initial one (2.19e-3) is deliberate: early on the data gradient
dominates by 100x regardless of lambda, so the penalty can only change the
outcome late.

| label | lambda1 |
|---|---|
| `0` | 0 (reference) |
| `7e-07` | 6.75e-7 |
| `2e-06` | 2.13e-6 |
| `7e-06` | 6.75e-6 |
| `2e-05` | 2.13e-5 (= lambda1\*) |
| `7e-05` | 6.75e-5 |
| `2e-04` | 2.13e-4 |
| `7e-04` | 6.75e-4 |

## Requirements

- MPI toolchain (OpenMPI `mpicxx` / `mpirun`). On this machine:
  `export PATH=/usr/lib64/openmpi/bin:$PATH`.
- A C++20 compiler.
- `dataset/cnn_dataset_train.npz` present — regenerate it with the runbook in
  [`dataset/README.md`](../../../dataset/README.md) if missing.
- All commands are run **from the repository root**.

## Build

```bash
export PATH=/usr/lib64/openmpi/bin:$PATH
mkdir -p build/experiments
mpicxx -std=c++20 -O3 -ICNN/src \
  CNN/experiments/regularization_tuning/main.cpp \
  CNN/src/core/*.cpp CNN/src/data/*.cpp CNN/src/layers/*.cpp \
  CNN/src/model/*.cpp CNN/src/optimizers/*.cpp \
  CNN/src/training/*.cpp CNN/src/tuning/*.cpp \
  -o build/experiments/regularization_tuning
```

## Run

> **Dataset prerequisite.** The reported results were obtained from a dataset
> regenerated with `python3 build_dataset.py --seed 42`. The pipeline correction
> (repository-relative paths, deterministic ordering, and an explicit seed) is
> a separate pull request and is not yet on `main`. The `.npz` files currently
> committed on `main` contain invalid data (Gaussian noise instead of Signed
> Distance Function values), so running this experiment without that correction
> produces meaningless results.

Smoke test first:

```bash
mpirun -n 8 --oversubscribe build/experiments/regularization_tuning converge 20
```

Then the convergence probe and the two sweeps. The sweeps are independent and
can run concurrently on a machine with enough cores:

```bash
mkdir -p results/cross_validation/regularization_tuning
mpirun -n 8 --oversubscribe build/experiments/regularization_tuning converge 500 \
  > results/cross_validation/regularization_tuning/convergence_fold0.csv
mpirun -n 8 --oversubscribe build/experiments/regularization_tuning sweep-l2 100
mpirun -n 8 --oversubscribe build/experiments/regularization_tuning sweep-l1 100
```

Each sweep writes one row per (candidate, fold) to
`results/cross_validation/regularization_tuning/sweep_{l1,l2}.csv`, with
columns `candidate, l1_weight, l2_weight, fold, train_mse, val_mse,
baseline_mse, l1_penalty, l2_penalty, sum_w2, weight_change_norm, epochs`.

Then run the paired analysis, which applies the decision rule:

```bash
python3 CNN/experiments/regularization_tuning/analyze.py \
  results/cross_validation/regularization_tuning/sweep_l2.csv l2
python3 CNN/experiments/regularization_tuning/analyze.py \
  results/cross_validation/regularization_tuning/sweep_l1.csv l1
```

## Results

Both sweeps ran on 8 MPI ranks, concurrently, ~123 min each under the reported
machine conditions. This is longer than the compute-only estimate because of
process, loading, and other wall-clock overhead.
`lambda = 0` reproduces the value measured independently in the diagnostics
(`0.00577535 +/- 0.00114692`), which confirms the fold plan and the seeds are
identical and the pairing is legitimate.

### Sweep A — L2 (Ridge)

| lambda2 | val MSE | std | val/baseline | **paired diff vs 0** | std | improved | L2 penalty | sum(w^2) | \|dw\| |
|---|---|---|---|---|---|---|---|---|---|
| **0** | **0.005776** | 0.001282 | 0.01161 | — | — | — | 0 | 341.47 | 0.48 |
| 1e-4 | 0.005812 | 0.001309 | 0.01168 | +0.0000360 | 0.0000709 | 3/5 | 0.030 | 301.99 | 2.85 |
| 3.16e-4 | 0.005826 | 0.001302 | 0.01171 | +0.0000496 | 0.0000535 | 2/5 | 0.085 | 269.40 | 4.37 |
| 1e-3 | 0.005817 | 0.001302 | 0.01170 | +0.0000411 | 0.0000492 | 2/5 | 0.226 | 225.64 | 6.16 |
| 3.16e-3 | 0.005896 | 0.001340 | 0.01185 | +0.0001196 | 0.0000646 | 0/5 | 0.562 | 177.73 | 8.04 |
| 1e-2 | 0.005937 | 0.001325 | 0.01194 | +0.0001612 | 0.0001245 | 0/5 | 1.352 | 135.18 | 9.69 |
| 3.16e-2 | 0.006206 | 0.001358 | 0.01248 | +0.0004300 | 0.0001845 | 0/5 | 3.269 | 103.43 | 10.97 |
| 1e-1 | 0.006730 | 0.001274 | 0.01353 | +0.0009543 | 0.0002532 | 0/5 | 8.576 | 85.76 | 11.85 |

**lambda2\* = 0.** All 7 non-zero values have a positive observed mean paired
difference, and none improves in 4 folds or more. The largest degradation is at
the largest lambda, but the intermediate values are noisy rather than strictly
monotonic.

### Sweep B — L1 (Lasso)

| lambda1 | val MSE | std | val/baseline | **paired diff vs 0** | std | improved | L1 penalty | sum(w^2) | \|dw\| |
|---|---|---|---|---|---|---|---|---|---|
| **0** | **0.005776** | 0.001282 | 0.01161 | — | — | — | 0 | 341.47 | 0.48 |
| 6.75e-7 | 0.005805 | 0.001294 | 0.01167 | +0.0000290 | 0.0000451 | 2/5 | 0.009 | 329.13 | 1.58 |
| 2.13e-6 | 0.005812 | 0.001297 | 0.01168 | +0.0000357 | 0.0000363 | 1/5 | 0.027 | 313.93 | 2.74 |
| 6.75e-6 | 0.005803 | 0.001305 | 0.01167 | +0.0000269 | 0.0000303 | 2/5 | 0.079 | 287.35 | 4.41 |
| 2.13e-5 | 0.005823 | 0.001312 | 0.01171 | +0.0000468 | 0.0000537 | 0/5 | 0.215 | 248.35 | 6.44 |
| 6.75e-5 | 0.005874 | 0.001376 | 0.01181 | +0.0000977 | 0.0001159 | 1/5 | 0.532 | 199.57 | 8.71 |
| 2.13e-4 | 0.005846 | 0.001427 | 0.01175 | +0.0000696 | 0.0001770 | 2/5 | 1.198 | 151.41 | 10.83 |
| 6.75e-4 | 0.006023 | 0.001434 | 0.01211 | +0.0002468 | 0.0001863 | 0/5 | 2.484 | 110.83 | 12.54 |

**lambda1\* = 0.** Same picture: all 7 non-zero values have a positive observed
mean paired difference, and none reaches 4 improving folds.

### Did lambda do anything? Yes

This is the control that stops the sweep from measuring the same run eight
times. `sum(w^2)` falls from 341.47 to 85.76 across the L2 grid (a 74.9%
spread) and to 110.83 across the L1 grid (67.5%), and `||w - w0||` grows from
0.48 to 11.85. The penalties are firmly in the active region — the grid is not
too low, so there was no reason to extend it upward.

If anything the opposite holds: even the smallest lambda tested already moves
`||w - w0||` by 6x. The whole grid is on the far side of the point where the
penalty starts to dominate, and the overall trend points toward 0 being the
optimum rather than toward a minimum inside the range.

### Verdict

**Regularization does not show a reliable benefit on this problem, and no
lambda is worth carrying into the group's final grid search.** The 14 non-zero
candidates all have positive observed mean paired differences versus lambda =
0, but none meets the predefined improvement rule. This matches what was
predicted from the diagnostics before running: with a train/validation gap of
+0.000413 against a 0.001397 fold spread, and validation MSE lower at epoch 500
than at epoch 100 without a sustained overfitting trend, there was no clear
excess capacity for a penalty to remove.

The 2D L1 x L2 grid is **not** justified and was not run.

A caveat on how small these effects are: the largest degradation measured, at
lambda2 = 0.1, is +0.00095 on a validation MSE of 0.0058 — about 0.7 times the
fold spread. Even the *harm* from over-regularizing is comparable with the
noise floor. The honest reading of both sweeps is that regularization is
irrelevant here, not that it is catastrophic.


## Caveat on absolute values

> I risultati sono ottenuti con lo split train/test casuale (seed 42), in cui la stessa
> geometria (profilo, angolo) compare in entrambi i set con Reynolds diverso. Poiché il
> Reynolds non influenza il target in modo misurabile, questi campioni sono di fatto
> duplicati fra train e test e le MSE di validazione riportate sono ottimistiche. I
> confronti *relativi* fra candidati restano validi perché tutti condividono lo stesso
> split; i valori *assoluti* andranno rimisurati se il gruppo adotterà uno split raggruppato.
