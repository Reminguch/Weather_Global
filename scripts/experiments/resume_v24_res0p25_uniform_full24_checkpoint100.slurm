#!/usr/bin/env bash
# Resume checkpoint 100 with streamed validation; preserve the 500-step schedule.
# Budget: 400 remaining updates at ~200-250 s plus compilation/validation.
# Keep 640G host RAM from job 13470357; 36h allows continuation overhead.
#SBATCH --job-name=v24-r025-uf24-resume
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=32
#SBATCH --mem=640G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/v24_r025_uf24_resume100_%j.out
#SBATCH --error=logs/v24_r025_uf24_resume100_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG="configs/experiments/v24_Ilya/res0p25_uniform_full24_legacy_di64_bcg2_bf16_dp4_lr1em5_sg500.json"
RUN_DIR="artifacts/checkpoints/v24_Ilya/res0p25_uniform_full24_di64_bcg2_dp4_20260905/legacy_di64_bcg2_uniform_full24_bf16_dp4_lr1em5_sg500"
CHECKPOINT="${RUN_DIR}/checkpoints/checkpoint_step00000100.pkl"
GPU_TELEMETRY="logs/v24_r025_uf24_resume100_${SLURM_JOB_ID}_gpu.csv"

[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing resume checkpoint: ${CHECKPOINT}" >&2; exit 1; }
exec 9>"${RUN_DIR}/resume.lock"
flock -n 9 || { echo "Another continuation holds the run lock" >&2; exit 1; }

export MPLCONFIGDIR="/tmp/mpl-v24-r025-uf24-di64-bcg2-dp4-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

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

echo "config=${CONFIG}"
echo "run_dir=${RUN_DIR}"
echo "resume=${CHECKPOINT}"
echo "temporal_init_scheme=legacy_haiku di=64 bcg=2 loss=all_steps_uniform_full24 lr=1e-5 precision=bf16 devices=4"
echo "git_commit=$(git rev-parse HEAD)"
git status --short
sha256sum \
  "${CONFIG}" \
  src/models/mamba/v24_Ilya/checkpoint.py \
  src/models/mamba/v24_Ilya/training/config.py \
  src/models/mamba/v24_Ilya/training/data.py \
  src/models/mamba/v24_Ilya/training/data_parallel.py \
  src/models/mamba/v24_Ilya/training/frame_data.py \
  src/models/mamba/v24_Ilya/training/endpoint_step.py \
  src/models/mamba/v24_Ilya/training/validation.py \
  src/models/mamba/v24_Ilya/training/runner.py

python -u scripts/training/train_v24_Ilya.py --config "${CONFIG}" --resume "${CHECKPOINT}"

GPU_PEAK_MIB="$(awk -F',' 'BEGIN { peak=0 } { gsub(/ /, "", $4); if (($4 + 0) > peak) peak=$4 + 0 } END { print peak }' "${GPU_TELEMETRY}")"
CPU_MAX_RSS="$(sstat -j "${SLURM_JOB_ID}.batch" --noheader --format=MaxRSS -P 2>/dev/null | tail -n 1 || true)"
echo "resource_summary sampled_gpu_peak_mib=${GPU_PEAK_MIB} cpu_max_rss=${CPU_MAX_RSS}"
