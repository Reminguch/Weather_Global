#!/usr/bin/env bash
# Five-update single-A100 smoke for the res0.25 v22-compatible all-24-step
# objective. This uses the production config but writes isolated artifacts and
# never submits evaluations or production training jobs.
#SBATCH --job-name=v24-r025-f24smk
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=288G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v24_r025_full24_smoke_%j.out
#SBATCH --error=logs/v24_r025_full24_smoke_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG="configs/experiments/v24_Ilya/res0p25_v22compat_all24_legacy_di16_bcg1_fp32_sg500.json"
OUTPUT_ROOT="artifacts/checkpoints/v24_Ilya/res0p25_v22compat_all24_loader_fix_smokes_20260904"
RUN_NAME="legacy_di16_bcg1_all24_no_tzero_fp32_lr3em6_smoke5_job${SLURM_JOB_ID}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
GPU_TELEMETRY="logs/v24_r025_full24_smoke_${SLURM_JOB_ID}_gpu.csv"

[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 1; }
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse existing run: ${RUN_DIR}" >&2; exit 1; }

export MPLCONFIGDIR="/tmp/mpl-v24-r025-full24-smoke-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

python - <<'PY'
import jax

devices = jax.local_devices()
print(f"jax_process_count={jax.process_count()}")
print(f"jax_local_devices={devices}")
if jax.process_count() != 1 or len(devices) != 1 or devices[0].platform != "gpu":
    raise SystemExit("expected one JAX process with exactly one GPU")
PY

echo "config=${CONFIG}"
echo "run_dir=${RUN_DIR}"
echo "objective=all_steps weights=24x1/24 ar_tail_k=20 max_steps=5"
echo "git_commit=$(git rev-parse HEAD)"
git status --short
sha256sum \
  "${CONFIG}" \
  src/models/mamba/v24_Ilya/checkpoint.py \
  src/models/mamba/v24_Ilya/training/config.py \
  src/models/mamba/v24_Ilya/training/data.py \
  src/models/mamba/v24_Ilya/training/frame_data.py \
  src/models/mamba/v24_Ilya/training/endpoint_step.py \
  src/models/mamba/v24_Ilya/training/runner.py

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

python -u scripts/training/train_v24_Ilya.py \
  --config "${CONFIG}" \
  --max-steps 5 \
  --checkpoint-every 5 \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}"

GPU_PEAK_MIB="$(awk -F',' 'BEGIN { peak=0 } { gsub(/ /, "", $4); if (($4 + 0) > peak) peak=$4 + 0 } END { print peak }' "${GPU_TELEMETRY}")"
CPU_MAX_RSS="$(sstat -j "${SLURM_JOB_ID}.batch" --noheader --format=MaxRSS -P 2>/dev/null | tail -n 1 || true)"
echo "resource_summary sampled_gpu_peak_mib=${GPU_PEAK_MIB} cpu_max_rss=${CPU_MAX_RSS}"

python -u scripts/training/verify_v24_Ilya_smoke.py \
  --metrics "${RUN_DIR}/train_metrics.jsonl" \
  --checkpoint "${RUN_DIR}/checkpoints/checkpoint_step00000005.pkl" \
  --expected-steps 5 \
  --max-median-step-seconds 180 \
  --gpu-peak-mib "${GPU_PEAK_MIB}" \
  --max-gpu-peak-mib $((72 * 1024))
