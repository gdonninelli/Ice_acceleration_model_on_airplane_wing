#!/bin/bash
#SBATCH --job-name=cnn-retraining
#SBATCH --account=EUHPC_D35_025
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=cnn_retraining_%j.out

set -euo pipefail

stage="${1:-}"
case "${stage}" in
    trial-1|trial-2|trial-3|trial-4|trial-4-seed-0|trial-4-seed-1|\
    trial-4-seed-0b|trial-4-seed-1b|trial-4-seed-42b|trial-5|production)
        ;;
    *)
        echo "Usage: sbatch $0 <stage>" >&2
        echo "Stages: trial-1 trial-2 trial-3 trial-4 trial-4-seed-0" >&2
        echo "        trial-4-seed-1 trial-4-seed-0b trial-4-seed-1b" >&2
        echo "        trial-4-seed-42b trial-5 production" >&2
        exit 2
        ;;
esac

# sbatch copies this script to a spool directory. The submission directory is
# the repository root for the plug-and-play commands documented below.
repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${repo_root}"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Stage: ${stage}"
echo "Repository: $(pwd)"
echo "Nodes: ${SLURM_JOB_NUM_NODES:-1}"
echo "MPI tasks: ${SLURM_NTASKS:-1}"

module purge
module load python/3.11.7
module load gcc/11.3.0
module load intel-oneapi-mpi

for dataset in dataset/cnn_dataset_train.npz dataset/cnn_dataset_test.npz; do
    if [[ ! -f "${dataset}" ]]; then
        echo "ERROR: $(pwd)/${dataset} is missing." >&2
        exit 1
    fi
done

build_dir="build/CNN-re-tuning-${SLURM_JOB_ID:-manual}"
cmake -S CNN -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target re_tuning --parallel

run_name="${stage}_slurm-${SLURM_JOB_ID:-manual}"
results_path="results/re-tuning/${run_name}"
if [[ -e "${results_path}" ]]; then
    echo "ERROR: ${results_path} already exists; refusing to overwrite it." >&2
    exit 1
fi

echo "Output: ${results_path}"
time mpirun -n "${SLURM_NTASKS:-1}" \
    "./${build_dir}/experiments/re_tuning" \
    --stage "${stage}" \
    --run-name "${run_name}" \
    --train-path dataset/cnn_dataset_train.npz \
    --test-path dataset/cnn_dataset_test.npz \
    --results-dir results \
    --diagnostics

# Submit one independent job per stage from the repository root:
# sbatch CNN/experiments/re_tuning/slurm.sh trial-1
# sbatch CNN/experiments/re_tuning/slurm.sh trial-2
# sbatch CNN/experiments/re_tuning/slurm.sh trial-3
# sbatch CNN/experiments/re_tuning/slurm.sh trial-4
# sbatch CNN/experiments/re_tuning/slurm.sh trial-4-seed-0
# sbatch CNN/experiments/re_tuning/slurm.sh trial-4-seed-1
# sbatch CNN/experiments/re_tuning/slurm.sh trial-4-seed-0b
# sbatch CNN/experiments/re_tuning/slurm.sh trial-4-seed-1b
# sbatch CNN/experiments/re_tuning/slurm.sh trial-4-seed-42b
# sbatch CNN/experiments/re_tuning/slurm.sh trial-5
# sbatch CNN/experiments/re_tuning/slurm.sh production
