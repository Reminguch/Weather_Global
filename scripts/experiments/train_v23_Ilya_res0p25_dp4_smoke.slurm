#!/usr/bin/env bash
# One single-node, four-A100 v23_Ilya data-parallel smoke. No automatic retry.
#SBATCH --job-name=v23-r025-dp4-smk
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=03:00:00
#SBATCH --output=logs/v23_r025_dp4_smoke_%j.out
#SBATCH --error=logs/v23_r025_dp4_smoke_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG="configs/experiments/v23_Ilya/res0p25_2019_2022_endpoint_k24_di16_bcg1_dp4_smoke.json"
RUN_NAME="di16_bcg1_last_step_closed_sg_stateful_dp4_5step_${SLURM_JOB_ID}"
GPU_TELEMETRY="logs/v23_r025_dp4_smoke_${SLURM_JOB_ID}_gpu_telemetry.csv"

export MPLCONFIGDIR="/tmp/mpl-v23-r025-dp4-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "config=${CONFIG}"
echo "run_name=${RUN_NAME}"
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

python -u scripts/training/train_v23_Ilya.py \
  --config "${CONFIG}" \
  --run-name "${RUN_NAME}"
