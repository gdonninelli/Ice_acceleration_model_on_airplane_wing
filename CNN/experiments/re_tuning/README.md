# Large-topology production retraining

This temporary experiment repeats the complete ordinary-training progression
with the topology selected by the architecture search:

```text
Conv2D: 8 channels, 5x5 kernel, stride 5, padding 0
Dense head: 1024, 512, 256, 128, 1
Activation: LeakyReLU, alpha 0.05
Optimizer: Adam, constant learning rate 1e-3
Physics weight: 0.10
Dropout: 0
L1/L2: 0/0
Gradient clipping: 1
```

Every stage starts from a fresh initialization. These are independent jobs,
not checkpoint continuations.

## Stages

| Stage | Batch construction | Batch size | Epoch cap | Stopping policy | Seed |
|---|---|---:|---:|---|---:|
| `trial-1` | range tail (`24x64 + 1x6`) | 64 | 200 | first ratio crossing | 42 |
| `trial-2` | balanced (`17x62 + 8x61`) | 64 | 200 | first ratio crossing | 42 |
| `trial-3` | balanced (`6x257`) | 257 | 200 | first ratio crossing | 42 |
| `trial-4` | balanced (`6x257`) | 257 | 834 | first ratio crossing | 42 |
| `trial-4-seed-0` | balanced (`6x257`) | 257 | 834 | first ratio crossing | 0 |
| `trial-4-seed-1` | balanced (`6x257`) | 257 | 834 | first ratio crossing | 1 |
| `trial-4-seed-0b` | balanced (`6x257`) | 257 | 834 | 20/20 policy | 0 |
| `trial-4-seed-1b` | balanced (`6x257`) | 257 | 834 | 20/20 policy | 1 |
| `trial-4-seed-42b` | balanced (`6x257`) | 257 | 834 | 20/20 policy | 42 |
| `trial-5` | balanced (`6x257`) | 257 | 1200 | 20/20 policy | 42 |
| `production` | balanced (`6x257`) | 257 | 2500 | 20/20 policy | 42 |

The first-crossing policy stops as soon as validation physical MSE is more
than 15 percent above training physical MSE. The 20/20 policy waits at least
20 epochs and requires 20 consecutive qualifying epochs without a new
validation minimum. Both policies restore the best validation checkpoint.

## Cluster submissions

Run these commands from the repository root. Wait for each scientific stage
to finish before interpreting the next one; jobs that differ only by seed may
run concurrently.

```bash
sbatch CNN/experiments/re_tuning/slurm.sh trial-1
sbatch CNN/experiments/re_tuning/slurm.sh trial-2
sbatch CNN/experiments/re_tuning/slurm.sh trial-3
sbatch CNN/experiments/re_tuning/slurm.sh trial-4
sbatch CNN/experiments/re_tuning/slurm.sh trial-4-seed-0
sbatch CNN/experiments/re_tuning/slurm.sh trial-4-seed-1
sbatch CNN/experiments/re_tuning/slurm.sh trial-4-seed-0b
sbatch CNN/experiments/re_tuning/slurm.sh trial-4-seed-1b
sbatch CNN/experiments/re_tuning/slurm.sh trial-4-seed-42b
sbatch CNN/experiments/re_tuning/slurm.sh trial-5
sbatch CNN/experiments/re_tuning/slurm.sh production
```

Each job uses an isolated `build/CNN-re-tuning-<job-id>` directory so jobs may
build concurrently, and writes to:

```text
results/re-tuning/<stage>_slurm-<job-id>/
```

The output directory contains `metadata.json`, `epoch_metrics.csv`, gradient,
update and activation diagnostics, `model_weights.bin`, `final_metrics.csv`,
and `test_metrics.csv`. The script refuses to overwrite an existing non-empty
run directory.

## Local smoke test

```bash
cmake -S CNN -B build/CNN -DCMAKE_BUILD_TYPE=Release
cmake --build build/CNN --target re_tuning --parallel
mpirun -n 2 ./build/CNN/experiments/re_tuning \
    --stage trial-1 \
    --run-name smoke-trial-1 \
    --smoke \
    --results-dir /tmp/cnn-re-tuning-smoke
```
