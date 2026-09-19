#!/usr/bin/env bash
# Width512 di16 batch1 reference13435248_7:10k26:11,224GiB host peak.
# 20k plus smoke/validation:about53h;72h/288G provide measured headroom.
# Evaluation overrides:gpu-medium,4h,20G for three exact32/40-lead artifacts.
#SBATCH --job-name=v24-r1-fresh512
#SBATCH --account=dabanin
#SBATCH --qos=gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=288G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/v24_r1_fresh512_%j.out
#SBATCH --error=logs/v24_r1_fresh512_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
STAGE="${1:?stage required}"
EXPERIMENT="${2:?absolute experiment directory required}"
export PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/mpl-v24-fresh-${SLURM_JOB_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source "${EXPERIMENT}/source/scripts/graphcast_env.sh"
export PYTHONPATH="${EXPERIMENT}/source:${EXPERIMENT}/source/third_party/graphcast"
GPU_LOG="${EXPERIMENT}/logs/${STAGE}_${SLURM_JOB_ID}_gpu.csv"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits -l 10 > "${GPU_LOG}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT
echo "stage=${STAGE} job=${SLURM_JOB_ID} experiment=${EXPERIMENT}"
python -u "${EXPERIMENT}/source/scripts/experiments/run_v24_res1_fresh_long.py" \
  "${STAGE}" --experiment "${EXPERIMENT}"
