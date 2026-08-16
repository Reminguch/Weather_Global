#!/usr/bin/env bash
# Four-A100 data-parallel v23 sparse-loss training. Submit as two exact-resume
# stages because 8,000 measured res0.25 sparse updates exceed one 6-day window.
# Four single-GPU smokes peaked at the 128G CPU limit; 640G leaves 1.25x
# aggregate headroom on the 1TB four-A100 nodes.
#SBATCH --job-name=v23-r025-sp3e6-dp4
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=32
#SBATCH --mem=640G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/v23_r025_sparse_lr3em6_dp4_8k_%j.out
#SBATCH --error=logs/v23_r025_sparse_lr3em6_dp4_8k_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG="configs/experiments/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp4_8k.json"
MAX_STEPS="${MAX_STEPS:-4000}"
RESUME="${RESUME:-}"
GPU_TELEMETRY="logs/v23_r025_sparse_lr3em6_dp4_8k_${SLURM_JOB_ID}_gpu.csv"

export MPLCONFIGDIR="/tmp/mpl-v23-r025-sp3e6-dp4-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "config=${CONFIG}"
echo "max_steps=${MAX_STEPS}"
echo "resume=${RESUME:-fresh_initialization}"
echo "gpu_telemetry=${GPU_TELEMETRY}"
echo "git_commit=$(git rev-parse HEAD)"
git status --short
sha256sum \
  "${CONFIG}" \
  src/models/mamba/v23_Ilya/checkpoint.py \
  src/models/mamba/v23_Ilya/training/config.py \
  src/models/mamba/v23_Ilya/training/data.py \
  src/models/mamba/v23_Ilya/training/data_parallel.py \
  src/models/mamba/v23_Ilya/training/endpoint_step.py \
  src/models/mamba/v23_Ilya/training/validation.py \
  src/models/mamba/v23_Ilya/training/runner.py

python - <<'PY'
import jax

devices = jax.local_devices()
print(f"jax_process_count={jax.process_count()}")
print(f"jax_local_device_count={len(devices)}")
for index, device in enumerate(devices):
    print(f"jax_device[{index}]={device} platform={device.platform} kind={device.device_kind}")
if jax.process_count() != 1 or len(devices) != 4:
    raise SystemExit("expected one JAX process with exactly four local devices")
if any(device.platform != "gpu" for device in devices):
    raise SystemExit("all four JAX devices must be GPUs")
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
  --checkpoint-every 1000
)
if [[ -n "${RESUME}" ]]; then
  if [[ ! -f "${RESUME}" ]]; then
    echo "Missing exact-resume checkpoint: ${RESUME}" >&2
    exit 2
  fi
  TRAIN_ARGS+=(--resume "${RESUME}")
fi

python -u scripts/training/train_v23_Ilya.py "${TRAIN_ARGS[@]}"

GPU_PEAK_MIB="$(awk -F',' 'BEGIN { peak=0 } { gsub(/ /, "", $4); if (($4 + 0) > peak) peak=$4 + 0 } END { print peak }' "${GPU_TELEMETRY}")"
CPU_MAX_RSS="$(sstat -j "${SLURM_JOB_ID}.batch" --noheader --format=MaxRSS -P 2>/dev/null | tail -n 1 || true)"
echo "resource_summary sampled_gpu_peak_mib=${GPU_PEAK_MIB} cpu_max_rss=${CPU_MAX_RSS}"
