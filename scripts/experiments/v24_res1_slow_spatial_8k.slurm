#!/usr/bin/env bash
# 8,000 updates: extrapolated 64-73h; allow 96h. Peak CPU RAM up to 235 GiB.
# Evaluation overrides: 16G/04:30:00 for three matched32 evaluations (~3h observed).
#SBATCH --job-name=v24-spatial-8k
#SBATCH --account=dabanin
#SBATCH --qos=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/v24_spatial_8k_%A_%a.out
#SBATCH --error=logs/v24_spatial_8k_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/mpl-v24-spatial-${SLURM_JOB_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh
sha256sum --quiet -c artifacts/checkpoints/v24_Ilya/res1_batch4_slow_spatial_20260916_8k/source/SHA256SUMS
python -u scripts/experiments/run_v24_res1_slow_spatial_8k.py "${1:?stage required}" --task "${SLURM_ARRAY_TASK_ID:-0}"
