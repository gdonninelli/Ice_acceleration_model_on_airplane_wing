# Weight-Regularization (L1/L2) Tuning at lr = 1e-3 — Code Only, Not Yet Run

## Status

**Code and smoke-tested, not run on the real grid.** This branch
(`feature/regularization-tuning-lr1e3`, based on `main`) prepares a re-run of
`CNN/experiments/regularization_tuning` (Giulio Donninelli's experiment, on
`main`) at the learning rate and topology `optimizer_comparison` /
`activation_tuning` / `physics_weight_tuning_lr1e3` settled on. The original
experiment is untouched; this lives in a separate experiment directory.
**The actual sweep (16 candidates) has not been run** — verification here was
limited to a build check and a 2-fold/2-epoch smoke test, per instruction;
the real run happens on CINECA Leonardo.

## Question

`CNN/experiments/regularization_tuning` selected lambda\* = 0 for both L1 and
L2 at lr = 1e-5 on the `{128,64}` topology (roughly 7.5e4 parameters over
1713 training samples), where the measured train/validation gap was
+0.000413 against a 0.001397 fold spread — essentially no overfitting for a
penalty to remove. At lr = 1e-3 on `{1024,512,256,128}` that premise does
not hold automatically. Computed from the layer shapes (conv(8,5,5,0) on a
150x150 input flattens to 30x30x8 = 7200 features, no scalar concatenation
in this blueprint): the first dense layer alone is 7200 x 1024 + 1024 ≈
7.37M parameters, and the full network is **≈8.06M parameters** over the
same 1713 samples — not the ~7.4M figure originally suggested for this
README, which does not reproduce from the architecture as defined here
(recomputed independently rather than carried over unchecked). Either way,
~8M parameters on 1713 samples is a regime where regularization can plausibly
matter for the first time in this experiment's history; see
`physics_weight_tuning_lr1e3`, which already showed this topology/lr leaves
the lazy-training regime and fits the data substantially better than the old
one.

**If the current L1 grid (max 6.75e-4) turns out too low** to show an effect
once real overfitting is observed, it should be widened — but that is a
follow-up decision for after the sweep runs, not something to do now.

## Topology, held fixed

`conv5x5-dense-1024-512-256-128`, LeakyReLU alpha = 0.05, Adam lr = 1e-3,
physics weight = 0.25, batch 64, 5 folds, seed 42 — copied verbatim from
`CNN/experiments/optimizer_comparison/main.cpp` so this stays comparable to
that experiment and to `activation_tuning` / `physics_weight_tuning_lr1e3`.
None of these are CLI options: this experiment holds everything but the
regularization axis constant, by design.

## Grids, identical to the original

Two independent one-dimensional sweeps, L2 alone and L1 alone, each with
lambda = 0 as its own reference row (so
`CNN/experiments/regularization_tuning/analyze.py` — unmodified — runs
against either sweep's aggregated CSV directly):

- **L2**: `{0, 1e-4, 3.16e-4, 1e-3, 3.16e-3, 1e-2, 3.16e-2, 1e-1}`
- **L1**: `{0, 6.75e-7, 2.13e-6, 6.75e-6, 2.13e-5, 6.75e-5, 2.13e-4, 6.75e-4}`

16 candidates total, but the lambda = 0 run is identical for both axes (same
`TrialConfig`), so the orchestrator runs it once and reuses the CSV for the
other axis — **15 unique training runs**, not 16.

## CLI: nothing that selects a candidate requires recompiling

Fold count, epochs, seed, batch size, dataset path, results directory, and
which candidate to run are all `main.cpp` command-line arguments, not
compiled-in constants:

```
regularization_tuning_lr1e3 --mode cv --axis <l1|l2> --lambda V
    [--epochs N] [--folds N] [--seed N] [--batch-size N]
    [--train-path PATH] [--results-dir PATH]
    [--diagnostics|--no-diagnostics] [--histogram-bins N] [--smoke]
```

One candidate per invocation, mirroring `physics_weight_tuning_lr1e3` and
`activation_tuning`. `--mode probe` (one fold, every-epoch validation) is
available for an epoch-budget check before committing to the full sweep, the
same discipline used there.

## Diagnostics: added from scratch

The original `regularization_tuning` has **no diagnostics wiring at all** —
this is not a matter of flipping an existing `--diagnostics` flag. The
wiring here is copied from `optimizer_comparison` /
`physics_weight_tuning_lr1e3`: `--diagnostics` populates
`training.diagnostics.*` and passes a `TrainingRunContext` into
`Trainer::fit`, writing the usual per-epoch files (`epoch_metrics.csv`,
`gradient_norms.csv`, `parameter_update_ratios.csv`,
`activation_statistics.csv`, `activation_histograms.csv`,
`learning_rate_steps.csv`, `metadata.json`) under
`<results-dir>/regularization_tuning_lr1e3/<axis>_<label>/candidate_000/fold_NNN/`.

## Segmented execution

`orchestrator.py` runs one MPI invocation per (axis, lambda) candidate, with
an idempotent resume state (`orchestrator_state.json`) and per-axis CSV
aggregation (`sweep_l1_lr1e3.csv`, `sweep_l2_lr1e3.csv`), mirroring
`physics_weight_tuning_lr1e3/orchestrator.py`. It also implements the
lambda = 0 reuse described above (`copy_zero_candidate`), so the shared
reference run is computed once.

## Verification done here (code only, per instruction)

- `main` builds clean from this branch.
- Smoke test: `--mode cv --axis l1 --lambda 2.13e-5 --smoke --diagnostics`
  (2 folds, 2 epochs) — exit code 0, all 6 expected diagnostics files
  present for both folds plus the run-level `metadata.json`.
- `git diff main -- CNN/experiments/regularization_tuning/
  results/cross_validation/regularization_tuning/` — empty: the original
  experiment and its committed results are untouched.
- **No epoch-budget probe and no full sweep were run.** The epoch budget
  (100, matching the original and the other `_lr1e3` experiments) is a
  starting assumption, not a verified choice, for this specific
  architecture/axis combination — worth a `--mode probe` check before
  committing the full grid on the cluster, the same way
  `physics_weight_tuning_lr1e3` did before its sweep.

## Cluster notes

See the repository-level cluster notes (compiler, dataset provisioning,
SLURM script) documented alongside this PR's description — they are common
to this experiment and `layer_tuning_lr1e3`, so they are not duplicated
per-experiment. In short: verify the C++20 build with the cluster's actual
`gcc/11.3.0` module before trusting this branch compiles there (not
empirically tested from this environment — see the PR description for what
was and wasn't verified), and the dataset must come from
`fix/data-pipeline-portability`'s `build_dataset.py --seed 42` (not yet
merged) for results to be comparable to every other sweep in this project.

## Cost, not yet measured on this architecture/axis

No timing measurement was taken for this experiment specifically (out of
scope for this task: code only, smoke test only). `physics_weight_tuning_lr1e3`
measured ~76 min/candidate (5 folds x 100 epochs) for the same topology and
learning rate, on the reference machine used for that experiment (not
Leonardo). Applying that figure as a rough starting point: 15 unique
candidates x ~76 min ≈ **19h**, but this should be re-measured on Leonardo
itself (different hardware, different MPI rank count) before being trusted
for scheduling — see the PR description for the explicit caveat.
