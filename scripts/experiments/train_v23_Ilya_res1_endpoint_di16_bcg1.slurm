#!/usr/bin/env bash
# Standalone v23_Ilya endpoint-only res1 reference run.
# Matches the validated res0.25 K=24 trajectory: four teacher-prefix steps,
# 20 closed-loop steps, five chunks per carried-state segment, last-step loss,
# and staged 8,000-step exact-resume operation.
#SBATCH --job-name=v23-r1-end-di16
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=2-06:00:00
#SBATCH --output=logs/v23_r1_endpoint_di16_%j.out
#SBATCH --error=logs/v23_r1_endpoint_di16_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG="configs/experiments/v23_Ilya/res1_endpoint_k24_di16_bcg1_closed_sg_stateful_20k.json"
MAX_STEPS="${MAX_STEPS:-8000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"
GPU_MEMORY_LOG="logs/v23_r1_endpoint_di16_${SLURM_JOB_ID}_gpu_memory.csv"

export MPLCONFIGDIR="/tmp/mpl-v23-r1-endpoint-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "config=${CONFIG}"
echo "max_steps=${MAX_STEPS}"
echo "checkpoint_every=${CHECKPOINT_EVERY}"
echo "gpu_memory_log=${GPU_MEMORY_LOG}"
echo "git_commit=$(git rev-parse HEAD)"
git status --short
sha256sum \
  "${CONFIG}" \
  src/models/graphcast/training/core/prepared_array.py \
  src/models/mamba/v23_Ilya/anchor_manifest.py \
  src/models/mamba/v23_Ilya/checkpoint.py \
  src/models/mamba/v23_Ilya/training/config.py \
  src/models/mamba/v23_Ilya/training/data.py \
  src/models/mamba/v23_Ilya/training/endpoint_step.py \
  src/models/mamba/v23_Ilya/training/runner.py

monitor_gpu() {
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    sleep 10
  done
}

monitor_gpu > "${GPU_MEMORY_LOG}" &
GPU_MONITOR_PID=$!

cleanup() {
  kill "${GPU_MONITOR_PID}" 2>/dev/null || true
  wait "${GPU_MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python -u scripts/training/train_v23_Ilya.py \
  --config "${CONFIG}" \
  --max-steps "${MAX_STEPS}" \
  --checkpoint-every "${CHECKPOINT_EVERY}"
