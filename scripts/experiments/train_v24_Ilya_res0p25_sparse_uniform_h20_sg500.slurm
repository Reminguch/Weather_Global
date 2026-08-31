#!/usr/bin/env bash
# Single-A100 res0.25 FP32-physical uniform-sparse-H20 ablation through 500 updates.
# This changes only the sparse horizon coefficients from [1,1,2,2,4,8] to uniform.
# Checkpoints are written every 50 updates for a dependent exact-GC screen.
# The heavy-tail sibling used 160 GB exactly, so this reserves 224 GB headroom.
#SBATCH --job-name=v24-r025-spuni
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=224G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v24_r025_spuni_%j.out
#SBATCH --error=logs/v24_r025_spuni_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG="configs/experiments/v24_Ilya/res0p25_sparse_uniform_legacy_di16_bcg1_fp32_sg500.json"
OUTPUT_ROOT="artifacts/checkpoints/v24_Ilya/res0p25_sparse_uniform_weight_ablation_20260829"
RUN_NAME="legacy_di16_bcg1_sparse_uniform_h20_no_tzero_fp32_lr3em6_sg500"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
MAX_STEPS=500
CHECKPOINT_EVERY=50

GPU_TELEMETRY="logs/v24_r025_spuni_${SLURM_JOB_ID}_gpu.csv"
[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 1; }
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse existing run: ${RUN_DIR}" >&2; exit 1; }

export MPLCONFIGDIR="/tmp/mpl-v24-r025-spuni-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

python - <<'PY'
import jax
devices = jax.local_devices()
print(f"jax_devices={devices}")
if jax.process_count() != 1 or len(devices) != 1 or devices[0].platform != "gpu":
    raise SystemExit("expected one JAX process with exactly one GPU")
PY

monitor_gpu() {
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
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
  --config "${CONFIG}" --max-steps "${MAX_STEPS}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --output-root "${OUTPUT_ROOT}" --run-name "${RUN_NAME}"
)
python -u scripts/training/train_v24_Ilya.py "${TRAIN_ARGS[@]}"

GPU_PEAK_MIB="$(awk -F',' 'BEGIN { peak=0 } { gsub(/ /, "", $4); if (($4 + 0) > peak) peak=$4 + 0 } END { print peak }' "${GPU_TELEMETRY}")"
CPU_MAX_RSS="$(sstat -j "${SLURM_JOB_ID}.batch" --noheader --format=MaxRSS -P 2>/dev/null | tail -n 1 || true)"
echo "resource_summary sampled_gpu_peak_mib=${GPU_PEAK_MIB} cpu_max_rss=${CPU_MAX_RSS}"

PADDED_STEP="$(printf '%08d' "${MAX_STEPS}")"
[[ -f "${RUN_DIR}/checkpoints/checkpoint_step${PADDED_STEP}.pkl" ]]
for step in 50 100 150 200 250 300 350 400 450 500; do
  [[ -f "${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl" ]]
done
