#!/usr/bin/env bash
# Five-update res1 compile/update smoke for the v24 FP32 physical trajectory.
# Prior res1 training peaked near 61 GiB host RAM; 128G allows for the FP32 tape.
#SBATCH --job-name=v24-r1-fp32-smk
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/v24_r1_fp32_smoke_%j.out
#SBATCH --error=logs/v24_r1_fp32_smoke_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG="configs/experiments/v24_Ilya/res1_sparse_h20_di16_bcg1_fp32_smoke.json"
RUN_DIR="artifacts/checkpoints/v24_Ilya/res1_fp32_smoke_20260826/di16_bcg1_sparse_h20_legacy_zero_head_fp32_lr3em6_5step"
GPU_TELEMETRY="logs/v24_r1_fp32_smoke_${SLURM_JOB_ID}_gpu.csv"

[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 1; }
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse existing run: ${RUN_DIR}" >&2; exit 1; }

export MPLCONFIGDIR="/tmp/mpl-v24-r1-fp32-smoke-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

monitor_gpu() {
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    sleep 5
  done
}
monitor_gpu > "${GPU_TELEMETRY}" &
GPU_MONITOR_PID=$!
cleanup() {
  kill "${GPU_MONITOR_PID}" 2>/dev/null || true
  wait "${GPU_MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python -u scripts/training/train_v24_Ilya.py --config "${CONFIG}"

GPU_PEAK_MIB="$(awk -F',' 'BEGIN { peak=0 } { gsub(/ /, "", $4); if (($4 + 0) > peak) peak=$4 + 0 } END { print peak }' "${GPU_TELEMETRY}")"
CPU_MAX_RSS="$(sstat -j "${SLURM_JOB_ID}.batch" --noheader --format=MaxRSS -P 2>/dev/null | tail -n 1 || true)"
echo "resource_summary sampled_gpu_peak_mib=${GPU_PEAK_MIB} cpu_max_rss=${CPU_MAX_RSS}"
[[ -f "${RUN_DIR}/checkpoints/checkpoint_step00000005.pkl" ]]

