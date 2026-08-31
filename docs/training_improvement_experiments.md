# Training Improvement Experiments

This document records the experiments used to investigate intermittent
physical-MSE spikes in the ordinary training run.

## Common Configuration

- Dataset: `dataset/cnn_dataset_train.npz` for training and the holdout split;
  `dataset/cnn_dataset_test.npz` is used only for final scoring.
- Seeds: `42` for the original runs; `0`, `1`, and `42` for the multi-seed
  reruns.
- MPI size: `64` ranks.
- Optimizer: Adam with learning rate `1e-3`, beta1 `0.9`, beta2 `0.999`.
- SIMM physics weight: `0.1`.
- Gradient clipping: existing element-wise threshold `1.0`; global-norm
  clipping has not been implemented yet.
- Architecture: Conv2D, LeakyReLU, flatten, scalar concatenation, and dense
  layers `128 -> 64 -> 1`.

The optimization objective remains the SIMM objective. Physical MSE is a
separate reporting and selection metric evaluated after each epoch's optimizer
updates.

## Initial Diagnosis

The original run had `1542` training samples and global batch size `64`:

- Each epoch performed `25` Adam updates.
- The first `24` batches contained `64` samples.
- The final batch contained only `6` samples.
- With `64` MPI ranks, that final update activated only six ranks.

The large physical-MSE spikes occurred when the final update also had the
largest gradient and parameter update of the epoch. Training and validation
MSE increased together, so this was a model-wide update perturbation rather
than validation overfitting.

The MPI reduction itself was not the suspected bug. Gradients are weighted by
each rank's sample count and globally averaged before the optimizer update.

## What Batch Balancing Means

Batch balancing distributes samples as evenly as possible across the fixed
number of training batches instead of putting every remainder sample into one
small final batch. It balances batch sizes only; it does not stratify airfoil
types, Reynolds numbers, or angles of attack.

For `1542` samples and a requested global batch size of `64`, the trainer first
keeps `25` batches because `ceil(1542 / 64) = 25`. It then distributes the
samples as:

```text
Without balancing: 64, 64, ..., 64, 6
With balancing:    17 batches of 62, followed by 8 batches of 61
```

Every sample is still used exactly once per epoch. No samples are dropped,
duplicated, or padded. The requested batch size is treated as an upper bound.
The range calculation is equivalent to:

```text
number_of_batches = ceil(samples / requested_batch_size)
base_size = samples / number_of_batches
remainder = samples % number_of_batches

first `remainder` batches have `base_size + 1` samples
remaining batches have `base_size` samples
```

The shuffled training order is created before these ranges are assigned, so the
sample composition changes each epoch while the batch sizes remain balanced.
Each balanced global batch is then partitioned across MPI ranks. With `64`
ranks, a global batch of `64` gives approximately one sample per rank, while a
global batch of `257` gives approximately four or five samples per rank.
The global gradient is still the average over the complete global batch; the
number of samples on an individual rank does not change that statistical batch
size when MPI synchronization is correct.

Balancing does not increase the total number of samples or the number of
optimizer updates. It replaces a high-variance six-sample update with updates
of roughly `61/62` samples. Since the noise of a sample mean scales roughly as
`1 / sqrt(batch_size)`, the six-sample update can have about
`sqrt(61 / 6) ~= 3.2` times the sampling noise of a balanced update.

For batch size `257`, the current training split divides exactly:

```text
1542 / 257 = 6 batches of 257
```

Therefore balancing has no additional effect in trial 3 beyond confirming that
there is no small tail. The main trial 3 change is the larger global batch and
the resulting reduction from `25` to `6` Adam updates per epoch.

## Experiment 1: Original Run

Result directory:
`results/ordinary-training/slurm-55367279_trial_1`

Configuration:

- Global batch size: `64`.
- Batches per epoch: `25`.
- Adam updates over 200 epochs: `5000`.
- Fixed-size batching with a final six-sample tail.

Observed behavior:

- Major training-MSE spikes occurred at epochs `84`, `93`, `102`, and `151`.
- The largest non-initial spike was approximately `0.01593` training MSE and
  `0.01649` validation MSE at epoch `93`.
- Final selected epoch: `195`.
- Final training, validation, and test physical MSE:
  `0.0022016`, `0.0018895`, and `0.0018235`.

Interpretation: the results were accurate overall, but some small final-batch
updates caused large temporary excursions.

## Experiment 2: Balanced Batches, Requested Size 64

Result directory:
`results/ordinary-training/slurm-55373639_trial_2`

The metadata still reports global batch size `64` and `25` batches. The code,
however, distributed the samples across balanced batches of approximately
`61/62` samples instead of using `24` full batches plus a six-sample tail.

Reason for the choice: test whether removing the pathological tail update was
enough to reduce the spikes without changing the nominal global batch size.

Observed behavior:

- The major three-to-four-fold spikes disappeared.
- Maximum non-initial training MSE decreased to approximately `0.00819`.
- Maximum final gradient decreased from `4.34` to `3.39`.
- Maximum final update ratio decreased from `0.0100` to `0.0055`.
- Final selected epoch: `190`.
- Final training, validation, and test physical MSE:
  `0.0024080`, `0.0019529`, and `0.0020304`.

Interpretation: balancing the batches supported the tail-batch hypothesis and
improved stability. The final test score was not better, so stability and
accuracy must be evaluated separately.

## Experiment 3: Global Batch Size 257

Result directory:
`results/ordinary-training/slurm-55375455_trial_3`

Configuration:

- Global batch size: `257`.
- Batches per epoch: `6`.
- The current `1542`-sample training split forms six exact global batches.
- Adam updates over 200 epochs: `1200`.

Reason for the choice: increase the global samples per update while retaining
more than one optimizer update per epoch and avoiding any small tail batch.

Observed behavior:

- The late recurrent spikes are effectively gone.
- A large early transient remains: training MSE was `0.05499` at epoch 1 and
  `0.05937` at epoch 2, then fell to `0.01873` at epoch 3.
- Only one clear post-initial excursion above twice the local neighboring
  baseline remained.
- Final selected epoch: `198`.
- Final training, validation, and test physical MSE:
  `0.0035720`, `0.0031052`, and `0.0026358`.

Interpretation: the larger batch removes the tail problem, but comparing 200
epochs directly with the batch-64 runs is not fair. It provides only about one
quarter as many Adam updates. The worse final score is therefore consistent
with undertraining, not evidence that batch size `257` is intrinsically worse.

For example, trial 3 reaches approximately `0.00776` training MSE after about
48 updates (epoch 8), while trial 2 reaches approximately `0.00807` after
about 50 updates (epoch 2). This supports comparing runs by optimizer-step
budget as well as by epoch count.

## Early-Stopping Follow-Up

The comparable-step trial was also checked with seeds `0` and `1`:

| Run | Epochs completed | Best epoch | Stop-time validation/training ratio | Test physical MSE |
| --- | ---: | ---: | ---: | ---: |
| `slurm-55383378_trial_4_seed_0` | 2 | 2 | `1.159144` | `0.017024867` |
| `slurm-55383296_trial_4_seed_1` | 4 | 1 | `1.197110` | `0.013266330` |
| `slurm-55377376_trial_4` (seed 42) | 527 | 470 | `1.159556` | `0.001923563` |

The original rule stopped at the first epoch where validation physical MSE
exceeded training physical MSE by more than `15%`. These runs show that a
single validation excursion could terminate training before the model had
completed its early transient. The seed 1 run also demonstrates why restoring
the best checkpoint matters: the selected epoch was 1 even though training
continued to epoch 4.

The stopping policy now uses these `TrainingConfig` defaults:

- `early_stopping_min_epochs = 20`.
- `early_stopping_patience = 20`.
- `max_overfit_ratio = 0.15` remains unchanged.

After the warm-up, an epoch counts toward the streak only when the validation
physical MSE exceeds the configured training-MSE ratio and does not improve
on the best validation checkpoint. Any validation improvement or return below
the ratio resets the streak. Training stops after 20 consecutive qualifying
epochs, and the best checkpoint is restored when configured. The SIMM
objective, optimizer, batch size, clipping, and validation split are unchanged.

### Rerun With the 20/20 Policy

The same `834`-epoch, six-batch configuration was rerun with the new stopping
policy:

| Seed | Run | Epochs completed | Stop epoch | Selected epoch | Final training MSE | Final validation MSE | Test MSE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `slurm-55390016_trial_4_seed_0b` | 834 | none | 743 | `0.001672786` | `0.001499316` | `0.001542168` |
| 1 | `slurm-55390186_trial_4_seed_1b` | 210 | 210 | 190 | `0.003171709` | `0.005345270` | `0.002699663` |
| 42 | `slurm-55391033_trial_4_b` | 834 | none | 831 | `0.001321096` | `0.001325427` | `0.001377287` |

The stopping behavior matches the intended policy:

- Seed 0 had 18 raw ratio crossings, but its longest qualifying streak was
  only one epoch, so it completed the full budget.
- Seed 1 had 206 raw ratio crossings. After its best checkpoint at epoch 190,
  epochs 191--210 formed 20 consecutive qualifying epochs, so training stopped
  at epoch 210 and restored epoch 190.
- Seed 42 had 74 raw ratio crossings, but its longest qualifying streak was
  four epochs, so it completed the full budget and selected epoch 831.

Compared with the previous first-crossing runs, the untouched test-set MSE
improved for every seed:

| Seed | Previous test MSE | New test MSE | Reduction |
| ---: | ---: | ---: | ---: |
| 0 | `0.017024867` | `0.001542168` | `90.94%` |
| 1 | `0.013266330` | `0.002699663` | `79.65%` |
| 42 | `0.001923563` | `0.001377287` | `28.40%` |

Across the three seeds, mean test MSE fell from `0.010738254` to
`0.001873040`, an `82.56%` reduction. The seed-to-seed range also narrowed
from `8.85x` between the best and worst runs to `1.96x`. The remaining
variation is largely validation-split variation because the split still uses
the training seed.

The early optimization transient remains, but it is finite and bounded after
the first two epochs. The largest post-transient training/validation MSE pairs
were `0.023183/0.024127` for seed 0, `0.027943/0.030847` for seed 1, and
`0.018733/0.018893` for seed 42. All numeric values in the new diagnostic CSVs
are finite. The fix therefore addresses premature termination and checkpoint
selection; it does not remove the initial optimizer transient.

### Final Seed-42 Training

The seed-42 configuration was then extended to a `2500`-epoch maximum:

Result directory:
`results/ordinary-training/slurm-final_training`

- Global batch size: `257`, giving six Adam updates per epoch.
- The run completed `1525` epochs, or `9150` optimizer updates, before the
  `20/20` early-stopping policy triggered.
- The best checkpoint was epoch `1505`, or `9030` optimizer updates. Epochs
  `1506--1525` formed the required 20-epoch qualifying streak.
- Final selected training, validation, and test physical MSE:
  `0.000493208`, `0.000599464`, and `0.000672714`.
- Test MSE by angle range was `0.000546783` for `|AoA| <= 10` degrees and
  `0.001365339` for `|AoA| > 10` degrees.

The selected test MSE is `51.16%` lower than the previous 834-epoch seed-42
run (`0.001377287`) and `63.11%` lower than the original trial 1 baseline
(`0.001823545`). The longer budget was therefore useful for this seed. The
final metrics equal the epoch-1505 checkpoint rather than the stopping epoch,
confirming that best-checkpoint restoration worked as intended.

## Comparable-Step Experiment

The planned experiment kept trial 3's configuration unchanged and increased the
epoch count to `834`:

- `834 * 6 = 5004` Adam updates.
- This closely matches the original batch-64 budget of `5000` updates.
- Keep MPI size `64`, global batch size `257`, learning rate `1e-3`, physics
  weight `0.1`, seed `42`, and the current element-wise clipping unchanged.

The Slurm command used:

```bash
--epochs 834 --batch-size 257 --learning-rate 1e-3
```

This isolated the effect of giving the larger-batch run a comparable optimizer
budget. The stopping-policy reruns show that the early transient is not caused
by premature stopping. Do not add global-norm clipping or change the learning
rate when comparing stopping policies. If reducing the remaining initial
excursion is still required, global L2 clipping should be evaluated as a
separate experiment.

## Clipping Status

The current `--gradient-clip 1` setting clips each gradient element
independently in `CNN/src/model/CNNModel.cpp`. It does not constrain the L2
norm of the complete network gradient. Diagnostics record raw synchronized
gradients before this clipping, while parameter-update diagnostics include the
actual clipped Adam update.

Global-norm clipping remains deferred so that its effect can be measured
independently from the completed batch-size and early-stopping experiments.

## Trial 3 Plot

The physical-MSE plot was generated with:

```bash
python3 CNN/analysis/plot_training_physical_mse.py \
  results/ordinary-training/slurm-55375455_trial_3/epoch_metrics.csv \
  --output results/ordinary-training/slurm-55375455_trial_3/physical_mse_comparison.png
```

Output: `results/ordinary-training/slurm-55375455_trial_3/physical_mse_comparison.png`.
