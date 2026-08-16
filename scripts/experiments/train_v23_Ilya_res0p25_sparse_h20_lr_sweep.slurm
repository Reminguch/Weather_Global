#!/usr/bin/env bash
# Four independent 200-update sparse-loss LR smokes on A100-80GB GPUs.
# Resource basis: prior res0.25 v23 single-GPU runs peaked near 60 GiB GPU and
# 83 GiB CPU RSS; six hours covers compile plus 200 steady-state updates.
#SBATCH --job-name=v23-r025-sparse-lr
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/v23_r025_sparse_lr_%A_%a.out
#SBATCH --error=logs/v23_r025_sparse_lr_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIGS=(
  "configs/experiments/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em7_smoke.json"
  "configs/experiments/v23_Ilya/res0p25_2019_2022_sparse_h20_lr1em6_smoke.json"
  "configs/experiments/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_smoke.json"
  "configs/experiments/v23_Ilya/res0p25_2019_2022_sparse_h20_lr1em5_smoke.json"
)
LR_LABELS=("3e-7" "1e-6" "3e-6" "1e-5")

TASK_ID="${SLURM_ARRAY_TASK_ID}"
if (( TASK_ID < 0 || TASK_ID >= ${#CONFIGS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
  exit 2
fi
CONFIG="${CONFIGS[TASK_ID]}"
LR_LABEL="${LR_LABELS[TASK_ID]}"
GPU_TELEMETRY="logs/v23_r025_sparse_lr_${SLURM_ARRAY_JOB_ID}_${TASK_ID}_gpu.csv"

export PROJECT_ROOT="${ROOT}"
export MPLCONFIGDIR="/tmp/mpl-v23-r025-sparse-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "array_job=${SLURM_ARRAY_JOB_ID} task=${TASK_ID} lr=${LR_LABEL}"
echo "config=${CONFIG}"
echo "gpu_telemetry=${GPU_TELEMETRY}"
echo "git_commit=$(git rev-parse HEAD)"
git status --short
sha256sum \
  "${CONFIG}" \
  src/models/mamba/v23_Ilya/checkpoint.py \
  src/models/mamba/v23_Ilya/training/config.py \
  src/models/mamba/v23_Ilya/training/data.py \
  src/models/mamba/v23_Ilya/training/frame_data.py \
  src/models/mamba/v23_Ilya/training/endpoint_step.py \
  src/models/mamba/v23_Ilya/training/runner.py

python - <<'PY'
import jax

devices = jax.local_devices()
print(f"jax_process_count={jax.process_count()}")
print(f"jax_local_device_count={len(devices)}")
for index, device in enumerate(devices):
    print(f"jax_device[{index}]={device} platform={device.platform} kind={device.device_kind}")
if jax.process_count() != 1 or len(devices) != 1:
    raise SystemExit("expected one JAX process with exactly one local device")
if devices[0].platform != "gpu":
    raise SystemExit("the sparse LR smoke requires one GPU")
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

python -u scripts/training/train_v23_Ilya.py --config "${CONFIG}"

GPU_PEAK_MIB="$(awk -F',' 'BEGIN { peak=0 } { gsub(/ /, "", $4); if (($4 + 0) > peak) peak=$4 + 0 } END { print peak }' "${GPU_TELEMETRY}")"
CPU_MAX_RSS="$(sstat -j "${SLURM_JOB_ID}.batch" --noheader --format=MaxRSS -P 2>/dev/null | tail -n 1 || true)"
echo "resource_summary sampled_gpu_peak_mib=${GPU_PEAK_MIB} cpu_max_rss=${CPU_MAX_RSS}"
