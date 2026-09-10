#!/usr/bin/env bash
# Training: comparable 13435248 tasks completed in 25.8–26.3h, up to 224GiB RSS.
# 36h and 288G allow compilation/preflight overhead and about 29% RAM headroom.
# Evaluation overrides: two exact32 artifacts, 2h and 20G, based on 13435335.
#SBATCH --job-name=v24-r1-width
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=288G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/v24_r1_width_%A_%a.out
#SBATCH --error=logs/v24_r1_width_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
STAGE="${1:?stage required}"
EXPERIMENT="${2:?absolute experiment directory required}"
export MPLCONFIGDIR="/tmp/mpl-v24-width-${SLURM_JOB_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
export PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
source "${EXPERIMENT}/source/scripts/graphcast_env.sh"
export PYTHONPATH="${EXPERIMENT}/source:${EXPERIMENT}/source/third_party/graphcast"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
GPU_LOG="${EXPERIMENT}/logs/${STAGE}_${SLURM_JOB_ID}_${TASK}_gpu.csv"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits -l 5 > "${GPU_LOG}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT
echo "stage=${STAGE} task=${TASK} job=${SLURM_JOB_ID} snapshot=${EXPERIMENT}/source"
python -u "${EXPERIMENT}/source/scripts/experiments/run_v24_res1_width128.py" \
  "${STAGE}" --experiment "${EXPERIMENT}" --task "${TASK}"
