#!/bin/bash
#SBATCH --job-name=graphcast-demo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/run_demo_gpu-%j.out
#SBATCH --error=logs/run_demo_gpu-%j.err
#
# Optional: set account if required on your cluster
# #SBATCH --account=dabanin

set -euo pipefail

PROJECT_ROOT=/scratch/gpfs/DABANIN/iv9432/Weather_global
cd "${PROJECT_ROOT}"
mkdir -p logs

echo "=== Job ${SLURM_JOB_ID} on $(hostname) ==="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo

source scripts/graphcast_env.sh

# Run the demo notebook on this GPU node. Output notebook will contain all results.
OUT_NB="logs/demo_gpu_${SLURM_JOB_ID}.ipynb"
jupyter nbconvert --execute \
  --to notebook \
  --output "$(basename "${OUT_NB}")" \
  --ExecutePreprocessor.timeout=7200 \
  --output-dir="$(dirname "${OUT_NB}")" \
  scripts/demo.ipynb

echo "Done. Executed notebook: ${OUT_NB}"
