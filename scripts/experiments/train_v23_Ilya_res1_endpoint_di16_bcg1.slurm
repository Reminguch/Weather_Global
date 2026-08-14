#!/usr/bin/env bash
# Standalone v23_Ilya endpoint-only res1 reference run.
# Matches v22-r1-di-bcg task 0 resources and model width while using the
# last-step-only objective and BF16 unique-frame weather tape.
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
GPU_MEMORY_LOG="logs/v23_r1_endpoint_di16_${SLURM_JOB_ID}_gpu_memory.csv"

export MPLCONFIGDIR="/tmp/mpl-v23-r1-endpoint-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "config=${CONFIG}"
echo "gpu_memory_log=${GPU_MEMORY_LOG}"
echo "git_commit=$(git rev-parse HEAD)"
git status --short
sha256sum \
  "${CONFIG}" \
  src/models/mamba/modules/temporal_mesh_mamba_Ilya.py \
  src/models/mamba/v23_Ilya/config.py \
  src/models/mamba/v23_Ilya/model.py \
  src/models/mamba/v23_Ilya/training/config.py \
  src/models/mamba/v23_Ilya/training/endpoint_bptt.py \
  src/models/mamba/v23_Ilya/training/step.py \
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

python -u scripts/training/train_v23_Ilya.py --config "${CONFIG}"
