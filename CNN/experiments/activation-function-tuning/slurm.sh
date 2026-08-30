#!/bin/bash
# Slurm submission script for the configured ordinary CNN training on CINECA.
# Replace site-specific placeholders before submitting with sbatch.

#SBATCH --job-name=cnn-training
#SBATCH --account=ACCOUNT_HERE
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --output=cnn_training_%j.out

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

module purge
module load python/3.11.7
module load gcc/11.3.0
module load intel-oneapi-mpi

if [[ ! -f dataset/cnn_dataset_train.npz ]]; then
    echo "ERROR: dataset/cnn_dataset_train.npz is missing." >&2
    exit 1
fi
if [[ ! -f dataset/cnn_dataset_test.npz ]]; then
    echo "ERROR: dataset/cnn_dataset_test.npz is missing." >&2
    exit 1
fi

cmake -S CNN -B build/CNN -DCMAKE_BUILD_TYPE=Release
cmake --build build/CNN --target cnn_executable --parallel

run_name="slurm-${SLURM_JOB_ID:-manual}"
time mpirun -n "${SLURM_NTASKS:-1}" \
    ./build/CNN/cnn_executable \
    --activation leakyrelu --alpha 0.05 \
    --epochs 200 --batch-size 64 --learning-rate 1e-3 \
    --physics-weight 0.10 --l1-weight 0 --l2-weight 0 \
    --dropout 0 --gradient-clip 1 --seed 42 \
    --train-path dataset/cnn_dataset_train.npz \
    --test-path dataset/cnn_dataset_test.npz \
    --results-dir results \
    --experiment ordinary-training \
    --run-name "${run_name}" \
    --diagnostics
