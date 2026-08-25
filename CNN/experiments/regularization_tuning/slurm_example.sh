#!/bin/bash
# Example SLURM submission script for the regularization_tuning sweep
# on CINECA Leonardo. This is a STARTING POINT, not a validated script: the
# group's actual SLURM scripts are not versioned in this repository, so confirm
# the module list, MPI launch line, account, partition, and wall time before
# submitting it.
#
# The executable evaluates both typed ParameterGrid searches in one C++/MPI
# process.
#
# TODO before submitting -- every value below marked TODO is a guess or a
# placeholder, not a verified setting:
#   - ACCOUNT / PARTITION: project-specific, unknown from this environment.
#   - MPI module: confirm the correct module name and version for Leonardo.
#   - --ntasks / --ntasks-per-node: re-measure on the target hardware.
#   - --time: the full in-process sweep has not been timed on this allocation.

#SBATCH --job-name=reg-tuning-bigarch
#SBATCH --account=TODO_ACCOUNT
#SBATCH --partition=TODO_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=16              # TODO: re-measure on Leonardo
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00          # TODO: validate full-sweep wall time
#SBATCH --output=reg_tuning_%j.out

set -euo pipefail

module purge
module load python/3.11.7       # Dataset reads NPZ arrays through Python/NumPy
module load gcc/11.3.0
# TODO: MPI module name and version are not verified.
module load TODO_MPI_MODULE

# --- Dataset ---
# build_dataset.py --seed 42 exists only on fix/data-pipeline-portability
# (PR #6, not yet merged into main as of this writing). Either regenerate the
# deterministic dataset from that branch or transfer the verified training NPZ.
if [ ! -f dataset/cnn_dataset_train.npz ]; then
    echo "ERROR: dataset/cnn_dataset_train.npz is missing." >&2
    exit 1
fi

# --- Build ---
cmake -S CNN -B build/CNN -DCMAKE_BUILD_TYPE=Release
cmake --build build/CNN --target regularization_tuning --parallel

# --- Complete C++ search ---
time mpirun -n "${SLURM_NTASKS}" \
    ./build/CNN/experiments/regularization_tuning \
    --epochs 100 --folds 5 --seed 42 \
    --diagnostics \
    --results-dir results/cross_validation/regularization_tuning
