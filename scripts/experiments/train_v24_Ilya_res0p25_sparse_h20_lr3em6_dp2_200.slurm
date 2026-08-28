#!/usr/bin/env bash
# Two-A100 FP32-physical-trajectory causal screen. Exact-resume checkpoints and
# one-segment fixed validation are written every 50 updates through step 200.
# Two single-GPU lanes reached the 128G CPU limit; 320G provides 1.25x aggregate
# headroom on the high-memory A100-80GB nodes.
#SBATCH --job-name=v24-r025-sp-dp2
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=08:00:00
#SBATCH --output=logs/v24_r025_sparse_lr3em6_dp2_200_%j.out
#SBATCH --error=logs/v24_r025_sparse_lr3em6_dp2_200_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG="${CONFIG:-configs/experiments/v24_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp2_200.json}"
MAX_STEPS="${MAX_STEPS:-200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
RUN_NAME="${RUN_NAME:-}"
RESUME="${RESUME:-}"
SMOKE_GATE="${SMOKE_GATE:-0}"
GPU_TELEMETRY="logs/v24_r025_sparse_lr3em6_dp2_200_${SLURM_JOB_ID}_gpu.csv"

export MPLCONFIGDIR="/tmp/mpl-v24-r025-sp-dp2-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "config=${CONFIG}"
echo "max_steps=${MAX_STEPS}"
echo "checkpoint_every=${CHECKPOINT_EVERY}"
echo "output_root=${OUTPUT_ROOT:-from_config}"
echo "run_name=${RUN_NAME:-from_config}"
echo "resume=${RESUME:-fresh_initialization}"
echo "smoke_gate=${SMOKE_GATE}"
echo "gpu_telemetry=${GPU_TELEMETRY}"
echo "git_commit=$(git rev-parse HEAD)"
git status --short
sha256sum \
  "${CONFIG}" \
  src/models/mamba/v24_Ilya/checkpoint.py \
  src/models/mamba/v24_Ilya/training/config.py \
  src/models/mamba/v24_Ilya/training/data.py \
  src/models/mamba/v24_Ilya/training/data_parallel.py \
  src/models/mamba/v24_Ilya/training/endpoint_step.py \
  src/models/mamba/v24_Ilya/training/validation.py \
  src/models/mamba/v24_Ilya/training/runner.py

python - <<'PY'
import jax

devices = jax.local_devices()
print(f"jax_process_count={jax.process_count()}")
print(f"jax_local_device_count={len(devices)}")
for index, device in enumerate(devices):
    print(f"jax_device[{index}]={device} platform={device.platform} kind={device.device_kind}")
if jax.process_count() != 1 or len(devices) != 2:
    raise SystemExit("expected one JAX process with exactly two local devices")
if any(device.platform != "gpu" for device in devices):
    raise SystemExit("both JAX devices must be GPUs")
PY

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

TRAIN_ARGS=(
  --config "${CONFIG}"
  --max-steps "${MAX_STEPS}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
)
if [[ -n "${OUTPUT_ROOT}" ]]; then
  TRAIN_ARGS+=(--output-root "${OUTPUT_ROOT}")
fi
if [[ -n "${RUN_NAME}" ]]; then
  TRAIN_ARGS+=(--run-name "${RUN_NAME}")
fi
if [[ -n "${RESUME}" ]]; then
  if [[ ! -f "${RESUME}" ]]; then
    echo "Missing exact-resume checkpoint: ${RESUME}" >&2
    exit 2
  fi
  TRAIN_ARGS+=(--resume "${RESUME}")
fi

python -u scripts/training/train_v24_Ilya.py "${TRAIN_ARGS[@]}"

GPU_PEAK_MIB="$(awk -F',' 'BEGIN { peak=0 } { gsub(/ /, "", $4); if (($4 + 0) > peak) peak=$4 + 0 } END { print peak }' "${GPU_TELEMETRY}")"
CPU_MAX_RSS="$(sstat -j "${SLURM_JOB_ID}.batch" --noheader --format=MaxRSS -P 2>/dev/null | tail -n 1 || true)"
echo "resource_summary sampled_gpu_peak_mib=${GPU_PEAK_MIB} cpu_max_rss=${CPU_MAX_RSS}"

if [[ "${SMOKE_GATE}" == "1" ]]; then
  if [[ -z "${OUTPUT_ROOT}" || -z "${RUN_NAME}" ]]; then
    echo "SMOKE_GATE=1 requires explicit OUTPUT_ROOT and RUN_NAME" >&2
    exit 3
  fi
  V24_SMOKE_STDERR="logs/v24_r025_sparse_lr3em6_dp2_200_${SLURM_JOB_ID}.err"
  if [[ -s "${V24_SMOKE_STDERR}" ]]; then
    echo "Smoke gate failed: stderr is not empty (${V24_SMOKE_STDERR})" >&2
    exit 3
  fi
  python -u scripts/training/verify_v24_Ilya_smoke.py \
    --metrics "${OUTPUT_ROOT}/${RUN_NAME}/train_metrics.jsonl" \
    --checkpoint "${OUTPUT_ROOT}/${RUN_NAME}/checkpoints/checkpoint_step00000005.pkl" \
    --expected-steps 5 \
    --max-median-step-seconds 85 \
    --gpu-peak-mib "${GPU_PEAK_MIB}" \
    --max-gpu-peak-mib $((72 * 1024)) \
    --require-loss-by-horizon
fi
