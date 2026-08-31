#!/usr/bin/env bash
# Four sequential FP32-physical res1 ablations, all trained through 10k:
# v22-compatible uniform -> sparse -> Mamba1 init -> no temporal output zero.
# The final residual projection stays zero-initialized in every lane.
#SBATCH --job-name=v24-r1-v22ab
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/v24_r1_v22ab_%A_%a.out
#SBATCH --error=logs/v24_r1_v22ab_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIGS=(
  "configs/experiments/v24_Ilya/res1_v22compat_uniform_legacy_di16_bcg1_fp32_10k.json"
  "configs/experiments/v24_Ilya/res1_v22compat_sparse_legacy_di16_bcg1_fp32_10k.json"
  "configs/experiments/v24_Ilya/res1_v22compat_sparse_mamba1_di16_bcg1_fp32_10k.json"
  "configs/experiments/v24_Ilya/res1_v22compat_sparse_mamba1_no_tzero_di16_bcg1_fp32_10k.json"
)
RUN_NAMES=(
  "v22compat_uniform_legacy_di16_bcg1_tzero_fp32_lr1em4_10k"
  "v22compat_sparse_legacy_di16_bcg1_tzero_fp32_lr1em4_10k"
  "v22compat_sparse_mamba1_di16_bcg1_tzero_fp32_lr1em4_10k"
  "v22compat_sparse_mamba1_di16_bcg1_no_tzero_fp32_lr1em4_10k"
)
OUTPUT_ROOT="artifacts/checkpoints/v24_Ilya/res1_v22compat_init_ablation_20260828"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
CONFIG="${CONFIGS[${TASK_ID}]}"
RUN_NAME="${RUN_NAMES[${TASK_ID}]}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
GPU_TELEMETRY="logs/v24_r1_v22ab_${SLURM_ARRAY_JOB_ID}_${TASK_ID}_gpu.csv"

[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 1; }
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse existing run: ${RUN_DIR}" >&2; exit 1; }

export MPLCONFIGDIR="/tmp/mpl-v24-r1-v22ab-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "task=${TASK_ID} config=${CONFIG} run_dir=${RUN_DIR}"
echo "git_commit=$(git rev-parse HEAD)"
git status --short
sha256sum "${CONFIG}" \
  src/models/mamba/v24_Ilya/model.py \
  src/models/mamba/v24_Ilya/training/data.py \
  src/models/mamba/v24_Ilya/training/endpoint_step.py \
  src/models/mamba/v24_Ilya/training/runner.py

monitor_gpu() {
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    sleep 10
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

for step in 2000 4000 6000 8000 10000; do
  [[ -f "${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl" ]]
done

python -u scripts/training/build_v24_Ilya_swa.py \
  --inputs \
    "${RUN_DIR}/checkpoints/checkpoint_step00002000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00004000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00006000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00008000.pkl" \
  --source-steps 2000 4000 6000 8000 \
  --output "${RUN_DIR}/swa/swa_step02000-08000.pkl"

GPU_PEAK_MIB="$(awk -F',' 'BEGIN { peak=0 } { gsub(/ /, "", $4); if (($4 + 0) > peak) peak=$4 + 0 } END { print peak }' "${GPU_TELEMETRY}")"
CPU_MAX_RSS="$(sstat -j "${SLURM_JOB_ID}.batch" --noheader --format=MaxRSS -P 2>/dev/null | tail -n 1 || true)"
echo "resource_summary sampled_gpu_peak_mib=${GPU_PEAK_MIB} cpu_max_rss=${CPU_MAX_RSS}"
