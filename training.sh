#!/bin/bash
#SBATCH --job-name=cnn-training
#SBATCH --account=EUHPC_D35_025
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --output=cnn_training_%j.out

set -euo pipefail

# Run from the directory where sbatch was submitted.
# This avoids Slurm executing the copied script from its spool directory.
repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${repo_root}"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Running from: $(pwd)"
echo "Nodes: ${SLURM_JOB_NUM_NODES:-1}"
echo "Tasks: ${SLURM_NTASKS:-1}"

module purge
module load python/3.11.7
module load gcc/11.3.0
module load intel-oneapi-mpi

if [[ ! -f dataset/cnn_dataset_train.npz ]]; then
    echo "ERROR: $(pwd)/dataset/cnn_dataset_train.npz is missing." >&2
    exit 1
fi

if [[ ! -f dataset/cnn_dataset_test.npz ]]; then
    echo "ERROR: $(pwd)/dataset/cnn_dataset_test.npz is missing." >&2
    exit 1
fi

echo "Datasets found:"
ls -lh dataset/cnn_dataset_train.npz
ls -lh dataset/cnn_dataset_test.npz

cmake -S CNN -B build/CNN -DCMAKE_BUILD_TYPE=Release
cmake --build build/CNN --target cnn_executable --parallel

run_name="slurm-${SLURM_JOB_ID:-manual}"

time mpirun -n "${SLURM_NTASKS:-1}" \
    ./build/CNN/cnn_executable \
    --activation leakyrelu \
    --alpha 0.05 \
    --epochs 834 \
    --batch-size 257 \
    --learning-rate 1e-3 \
    --physics-weight 0.10 \
    --l1-weight 0 \
    --l2-weight 0 \
    --dropout 0 \
    --gradient-clip 1 \
    --seed 42 \
    --train-path dataset/cnn_dataset_train.npz \
    --test-path dataset/cnn_dataset_test.npz \
    --results-dir results \
    --experiment ordinary-training \
    --run-name "${run_name}" \
    --diagnostics