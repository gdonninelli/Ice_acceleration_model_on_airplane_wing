#!/bin/bash
# Example SLURM submission script for the regularization_tuning_v2 sweep
# on CINECA Leonardo. This is a STARTING POINT, not a validated script: the
# group's actual SLURM scripts (used to produce
# results/dropout_tuning/dropout_sweep_53196963.out and
# results/cross_validation/layer_tuning/run_leonardo_51807287.log) are not
# versioned anywhere in this repository, so this was written from the
# project's CLI/orchestrator conventions plus the module names visible in
# those job logs (`python/3.11.7`, `gcc/11.3.0`) -- not copied from a working
# script. Ask Vittorio or Alessia for their scripts before relying on this
# one; at minimum, diff the module list and MPI launch line against theirs.
#
# TODO before submitting -- every value below marked TODO is a guess or a
# placeholder, not a verified setting:
#   - ACCOUNT / PARTITION: project-specific, unknown from this environment.
#   - MPI module: no MPI module load is visible in the group's existing job
#     logs (they may load it implicitly via the gcc module, or via a
#     Leonardo-specific toolchain module). Confirm the correct module name
#     (Leonardo commonly uses a spack-style name like
#     "openmpi/<version>--gcc--11.3.0", but this is not verified here).
#   - --ntasks / --ntasks-per-node: this project's own orchestrators default
#     to 4 MPI ranks per invocation (activation_tuning's comment: "measured
#     optimum on the reference machine, the workload is memory bound and 8
#     or 12 do not help"). That measurement was NOT redone here or on
#     Leonardo -- re-measure rather than trust it on different hardware.
#   - --time: NOT measured for this experiment (see README.md "Cost, not yet
#     measured on this architecture/axis"). Do not submit the full array
#     with a guessed walltime; time one candidate first.

#SBATCH --job-name=reg-tuning-v2
#SBATCH --account=TODO_ACCOUNT
#SBATCH --partition=TODO_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=4                 # TODO: re-measure; 4 is this project's default, not verified on Leonardo
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00            # TODO: PLACEHOLDER. Time one candidate (see below) before trusting this.
#SBATCH --output=reg_tuning_v2_%A_%a.out
#SBATCH --array=0                  # TODO: expand once per-candidate cost is known (see array note below)

set -euo pipefail

module purge
module load python/3.11.7
module load gcc/11.3.0
# TODO: MPI module -- name and version not verified, see note above.
module load TODO_MPI_MODULE

# --- Dataset ---
# build_dataset.py --seed 42 exists only on fix/data-pipeline-portability
# (PR #6, not yet merged into main as of this writing). Either:
#   (a) cherry-pick/merge that branch here and run
#       python3 build_dataset.py --seed 42
#       to regenerate dataset/cnn_dataset_{train,test}.npz on the cluster, or
#   (b) transfer the exact .npz files already verified against
#       dataset/CHECKSUMS.txt (sha256 90550654...59697e3 for train,
#       43a66940...86efa0b4 for test) from a machine that has them.
# Without one of these two, the cluster's fold split will not match the one
# used everywhere else in this project, and results will not be comparable.
if [ ! -f dataset/cnn_dataset_train.npz ] || [ ! -f dataset/cnn_dataset_test.npz ]; then
    echo "ERROR: dataset/*.npz missing. See the comment above this check." >&2
    exit 1
fi

# --- Build ---
cmake -S CNN -B build/CNN -DCMAKE_BUILD_TYPE=Release
cmake --build build/CNN --target regularization_tuning_v2 --parallel

# --- Cost probe: measure ONE candidate before the array is widened ---
# Uncomment and run this by itself first (single job, not an array):
# time mpirun -n ${SLURM_NTASKS} ./build/CNN/experiments/regularization_tuning_v2 \
#     --mode cv --axis l1 --lambda 2.13e-5 --epochs 100 --folds 5 \
#     --diagnostics --results-dir results/cross_validation/regularization_tuning_v2
# Multiply by 15 (unique candidates -- lambda=0 is shared between axes, see
# README.md) to project the full-sweep cost, and only then decide --array
# range and --time above.

# --- Segmented sweep, resumable ---
# The orchestrator itself launches one mpirun per candidate; running it
# under a single SLURM allocation with --ntasks matching --ranks below
# processes candidates sequentially within this job. Splitting into a real
# SLURM array (one candidate per array task) is the alternative if the
# per-candidate cost makes a single long job impractical -- not set up here
# since the per-candidate cost isn't known yet.
python3 CNN/experiments/regularization_tuning_v2/orchestrator.py \
    --ranks ${SLURM_NTASKS} \
    --epochs 100 --folds 5 --seed 42 --diagnostics \
    --results-dir results/cross_validation/regularization_tuning_v2
