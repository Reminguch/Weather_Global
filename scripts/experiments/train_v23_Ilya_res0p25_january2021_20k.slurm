#!/usr/bin/env bash
# Continue the validated v23_Ilya res0.25 January K=24 trajectory.
# No spatial/model sharding: host-owned weather/state tapes and one-step remat.
# The gpu QoS permits at most six days, so submit this as an exact-resume
# 8,000 -> 16,000 -> 20,000 dependency chain.
#SBATCH --job-name=v23-r025-jan-20k
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/v23_r025_jan_20k_%j.out
#SBATCH --error=logs/v23_r025_jan_20k_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG="configs/experiments/v23_Ilya/res0p25_january2021_endpoint_k24_di16_bcg1_smoke.json"
RUN_DIR="artifacts/checkpoints/v23_Ilya/res0p25_january2021_endpoint_k24_smoke_20260814_retry2/di16_bcg1_last_step_closed_sg_stateful_5step"
RESUME="${RESUME:-${RUN_DIR}/checkpoints/checkpoint_step00000005.pkl}"
MAX_STEPS="${MAX_STEPS:-8000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
GPU_MEMORY_LOG="logs/v23_r025_jan_20k_${SLURM_JOB_ID}_gpu_memory.csv"

if [[ ! -f "${RESUME}" ]]; then
  echo "Missing exact-resume checkpoint: ${RESUME}" >&2
  exit 2
fi

export MPLCONFIGDIR="/tmp/mpl-v23-r025-jan-20k-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "config=${CONFIG}"
echo "resume=${RESUME}"
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
    sleep 5
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
  --resume "${RESUME}" \
  --max-steps "${MAX_STEPS}" \
  --checkpoint-every "${CHECKPOINT_EVERY}"

GPU_PEAK_MIB="$(awk -F',' 'BEGIN { peak=0 } { gsub(/ /, "", $4); if (($4 + 0) > peak) peak=$4 + 0 } END { print peak }' "${GPU_MEMORY_LOG}")"
CPU_MAX_RSS="$(sstat -j "${SLURM_JOB_ID}.batch" --noheader --format=MaxRSS -P 2>/dev/null | tail -n 1 || true)"
echo "resource_summary sampled_gpu_peak_mib=${GPU_PEAK_MIB} cpu_max_rss=${CPU_MAX_RSS}"
