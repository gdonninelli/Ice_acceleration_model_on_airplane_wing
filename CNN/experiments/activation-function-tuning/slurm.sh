#!/bin/bash
# Slurm submission script for activation-function-tuning on CINECA.
# Replace site-specific placeholders before submitting with sbatch.

#SBATCH --job-name=activation-tuning
#SBATCH --account=ACCOUNT_HERE
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --output=activation_tuning_%j.out

set -euo pipefail

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

mkdir -p build/experiments
source_revision="$(git rev-parse --short=12 HEAD 2>/dev/null || printf unknown)"
mpicxx -std=c++20 -O3 -ICNN/src \
    -DCNN_SOURCE_REVISION=\"${source_revision}\" \
    CNN/experiments/activation-function-tuning/main.cpp \
    CNN/src/core/*.cpp CNN/src/data/*.cpp CNN/src/layers/*.cpp \
    CNN/src/model/*.cpp CNN/src/optimizers/*.cpp \
    CNN/src/training/*.cpp CNN/src/tuning/*.cpp \
    -o build/experiments/activation-function-tuning

time mpirun -n "${SLURM_NTASKS}" \
    ./build/experiments/activation-function-tuning \
    --epochs 100 --folds 5 --batch-size 64 --seed 42 \
    --train-path dataset/cnn_dataset_train.npz \
    --test-path dataset/cnn_dataset_test.npz \
    --results-dir results/cross_validation/activation-function-tuning \
    --diagnostic
