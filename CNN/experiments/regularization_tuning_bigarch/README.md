# Weight-Regularization (L1/L2) Tuning, Layer-Tuning Architecture — Code Only, Not Yet Run

## Status

**Code and smoke-tested, not run on the real grid.** This branch
(`feature/regularization-tuning-v2`, based on `main`) prepares a re-run of
`CNN/experiments/regularization_tuning` (Giulio Donninelli's experiment, on
`main`) on the `conv5x5-dense-1024-512-256-128` topology
(`optimizer_comparison` / `layer_tuning`'s winner), at the **project's
original learning rate, `1e-5`** -- unchanged from the original experiment.
The original experiment is untouched; this lives in a separate experiment
directory. **The actual sweep (16 candidates) has not been run** --
verification here was limited to a build check and a 2-fold/2-epoch smoke
test, per instruction; the real run happens on CINECA Leonardo.

**Naming/history note.** This branch was first prepared as
`regularization_tuning_lr1e3` (learning rate `1e-3`, matching
`physics_weight_tuning_lr1e3`), following an earlier group decision to move
every experiment to `lr = 1e-3`. That decision was **reversed**: the group
settled on `lr = 1e-5` everywhere except `physics_weight_tuning_lr1e3`
itself, which keeps `1e-3` because it already has real results measured
there and is the reason the learning-rate question was raised in the first
place. The directory was then renamed a second time, from
`regularization_tuning_v2` to `regularization_tuning_bigarch`: "v2"
implied a public first version that never existed (this branch was never
pushed), and `_bigarch` says what actually differs from the original
(a fixed, larger architecture), matching `layer_tuning_grid`'s naming
principle even though the literal suffix differs. **The branch name itself
is still `feature/regularization-tuning-v2` at the time of writing** --
renaming it to match was blocked by a pre-existing, unrelated local branch
named `feature/regularization-tuning`, and is pending a decision on that
branch before the rename can happen; update this note once it does.

## Question

`CNN/experiments/regularization_tuning` selected lambda\* = 0 for both L1 and
L2 at lr = 1e-5 on the `{128,64}` topology (roughly 7.5e4 parameters over
1713 training samples), where the measured train/validation gap was
+0.000413 against a 0.001397 fold spread -- essentially no overfitting for a
penalty to remove. **At lr = 1e-5, lambda\* = 0 is still the expected
outcome here too**, and for the same underlying reason: `1e-5` keeps the
network close to its initialization for the whole training budget (measured
weight-update ratio ~3.9e-5 in `physics_weight_tuning_lr1e3`'s README, on
the old topology -- the update-ratio mechanism is a property of the learning
rate, not of the architecture). A network that barely moves does not
overfit, so there is nothing for a weight penalty to correct, regardless of
how many parameters the network has on paper.

What *is* new relative to the original `regularization_tuning`: this
experiment runs on `{1024,512,256,128}` instead of `{128,64}` -- roughly
**8.06M parameters** instead of ~7.5e4 (computed from the layer shapes:
conv(8,5,5,0) on a 150x150 input flattens to 30x30x8 = 7200 features, no
scalar concatenation in this blueprint, so the first dense layer alone is
7200 x 1024 + 1024 ~ 7.37M). A ~100x larger network on the same 1713
samples, even one that moves very little per step, is a legitimately
different regime from the original's ~7.5e4-parameter network, and the
group's rationale for re-running this sweep at all is to check whether
capacity alone (independent of learning rate) changes the answer. The
honest expectation going in is still lambda\* = 0 -- update ratio, not
capacity, is what determines whether a network overfits in a fixed epoch
budget -- but "the network has 100x more capacity" is not nothing, so
running the sweep rather than assuming the answer is the point.

**If the current L1 grid (max 6.75e-4) turns out too low** to show an effect,
it should be widened -- but that is a follow-up decision for after the sweep
runs, not something to do now.

## Topology, held fixed

`conv5x5-dense-1024-512-256-128`, LeakyReLU alpha = 0.05, **Adam lr = 1e-5**,
physics weight = 0.25, batch 64, 5 folds, seed 42. Architecture copied
verbatim from `CNN/experiments/optimizer_comparison/main.cpp`; learning rate
kept at the project's original value, unchanged from
`CNN/experiments/regularization_tuning`. None of these are CLI options: this
experiment holds everything but the regularization axis constant, by
design.

## Grids, identical to the original

Two independent one-dimensional sweeps, L2 alone and L1 alone, each with
lambda = 0 as its own reference row (so
`CNN/experiments/regularization_tuning/analyze.py` -- unmodified -- runs
against either sweep's aggregated CSV directly):

- **L2**: `{0, 1e-4, 3.16e-4, 1e-3, 3.16e-3, 1e-2, 3.16e-2, 1e-1}`
- **L1**: `{0, 6.75e-7, 2.13e-6, 6.75e-6, 2.13e-5, 6.75e-5, 2.13e-4, 6.75e-4}`

16 candidates total, but the lambda = 0 run is identical for both axes (same
`TrialConfig`), so the orchestrator runs it once and reuses the CSV for the
other axis -- **15 unique training runs**, not 16.

## CLI: nothing that selects a candidate requires recompiling

Fold count, epochs, seed, batch size, dataset path, results directory, and
which candidate to run are all `main.cpp` command-line arguments, not
compiled-in constants:

```
regularization_tuning_bigarch --mode cv --axis <l1|l2> --lambda V
    [--epochs N] [--folds N] [--seed N] [--batch-size N]
    [--train-path PATH] [--results-dir PATH]
    [--diagnostics|--no-diagnostics] [--histogram-bins N] [--smoke]
```

One candidate per invocation, mirroring `physics_weight_tuning_lr1e3` and
`activation_tuning`. `--mode probe` (one fold, every-epoch validation) is
available for an epoch-budget check before committing to the full sweep, the
same discipline used there.

## Diagnostics: added from scratch

The original `regularization_tuning` has **no diagnostics wiring at all** --
this is not a matter of flipping an existing `--diagnostics` flag. The
wiring here is copied from `optimizer_comparison` /
`physics_weight_tuning_lr1e3`: `--diagnostics` populates
`training.diagnostics.*` and passes a `TrainingRunContext` into
`Trainer::fit`, writing the usual per-epoch files (`epoch_metrics.csv`,
`gradient_norms.csv`, `parameter_update_ratios.csv`,
`activation_statistics.csv`, `activation_histograms.csv`,
`learning_rate_steps.csv`, `metadata.json`) under
`<results-dir>/regularization_tuning_bigarch/<axis>_<label>/candidate_000/fold_NNN/`.
At lr = 1e-5 these diagnostics matter more, not less, than they would at a
higher rate: if the update ratios here turn out much larger than
`physics_weight_tuning_lr1e3`'s lr=1e-5 figure despite the identical
learning rate, that would indicate the larger architecture itself changes
the training dynamics independent of lr -- worth checking against, not
assuming away.

## Segmented execution

`orchestrator.py` runs one MPI invocation per (axis, lambda) candidate, with
an idempotent resume state (`orchestrator_state.json`) and per-axis CSV
aggregation (`sweep_l1_bigarch.csv`, `sweep_l2_bigarch.csv`), mirroring
`physics_weight_tuning_lr1e3/orchestrator.py`. It also implements the
lambda = 0 reuse described above (`copy_zero_candidate`), so the shared
reference run is computed once.

## Verification done here (code only, per instruction)

- `main` builds clean from this branch.
- Smoke test: `--mode cv --axis l1 --lambda 2.13e-5 --smoke --diagnostics`
  (2 folds, 2 epochs) -- exit code 0, all 6 expected diagnostics files
  present for both folds plus the run-level `metadata.json`.
- `git diff main -- CNN/experiments/regularization_tuning/
  results/cross_validation/regularization_tuning/` -- empty: the original
  experiment and its committed results are untouched.
- **No epoch-budget probe and no full sweep were run.** The epoch budget
  (100, matching the original) is a starting assumption, not a verified
  choice, for this specific architecture/axis combination -- worth a
  `--mode probe` check before committing the full grid on the cluster.

## Cluster notes

See the repository-level cluster notes (compiler, dataset provisioning,
SLURM script) documented alongside this PR's description -- they are common
to this experiment and `layer_tuning_grid`, so they are not duplicated
per-experiment. In short: the C++20 build was verified with the cluster's
actual `gcc/11.3.0` (via a local Docker container, not on Leonardo itself --
see the PR description), and the dataset must come from
`fix/data-pipeline-portability`'s `build_dataset.py --seed 42` (not yet
merged) for results to be comparable to every other sweep in this project.

## Cost, not yet measured on this architecture/learning-rate combination

No timing measurement was taken for this experiment specifically (out of
scope for this task: code only, smoke test only). `physics_weight_tuning_lr1e3`
measured ~76 min/candidate (5 folds x 100 epochs) for the same
**architecture**, but at `lr = 1e-3`, not `1e-5` -- per-epoch wall time
should be close either way (the learning rate doesn't change the amount of
arithmetic per step), but this has not been confirmed for `1e-5`
specifically. Applying that figure as a rough starting point anyway: 15
unique candidates x ~76 min ~ **19h**, but this should be re-measured on
Leonardo itself (different hardware, different MPI rank count, unconfirmed
lr-independence of per-epoch cost) before being trusted for scheduling.
