#!/usr/bin/env bash
# Submit with SHAPE=di16 (12 tasks) or SHAPE=di64 (12 tasks). The array is
# intentionally unthrottled; Slurm controls launch order from priority/resources.
#SBATCH --job-name=v24-r1-opt
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=224G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/v24_r1_opt_%A_%a.out
#SBATCH --error=logs/v24_r1_opt_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

SHAPE="${SHAPE:?Submit with --export=ALL,SHAPE=di16 or SHAPE=di64}"
case "${SHAPE}" in
  di16) CONFIG_GLOB="*_di16_bcg1_*.json" ;;
  di64) CONFIG_GLOB="*_di64_bcg4_*.json" ;;
  *) echo "Unsupported SHAPE=${SHAPE}; expected di16 or di64" >&2; exit 2 ;;
esac

CONFIG_DIR="configs/experiments/v24_Ilya/res1_optimizer_matrix_20260904"
mapfile -t CONFIGS < <(find "${CONFIG_DIR}" -maxdepth 1 -type f -name "${CONFIG_GLOB}" | sort)
[[ "${#CONFIGS[@]}" -eq 12 ]] || {
  echo "Expected 12 ${SHAPE} configs, found ${#CONFIGS[@]}" >&2
  exit 1
}
CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"
RUN_NAME="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"]["run_name"])' "${CONFIG}")"
RUN_DIR="artifacts/checkpoints/v24_Ilya/res1_optimizer_matrix_20260904/${RUN_NAME}"
GPU_TELEMETRY="logs/v24_r1_opt_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_gpu.csv"

[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 1; }
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse existing run: ${RUN_DIR}" >&2; exit 1; }

export MPLCONFIGDIR="/tmp/mpl-v24-r1-opt-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "shape=${SHAPE} task=${SLURM_ARRAY_TASK_ID} config=${CONFIG} run_dir=${RUN_DIR}"
echo "git_commit=$(git rev-parse HEAD)"
git status --short
sha256sum "${CONFIG}" \
  src/models/mamba/v24_Ilya/training/config.py \
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
    "${RUN_DIR}/checkpoints/checkpoint_step00003000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00004000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00005000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00006000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00007000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00008000.pkl" \
  --source-steps 2000 3000 4000 5000 6000 7000 8000 \
  --output "${RUN_DIR}/swa/swa_step02000-08000.pkl"

python -u scripts/training/build_v24_Ilya_swa.py \
  --inputs \
    "${RUN_DIR}/checkpoints/checkpoint_step00004000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00005000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00006000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00007000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00008000.pkl" \
  --source-steps 4000 5000 6000 7000 8000 \
  --output "${RUN_DIR}/swa/swa_step04000-08000.pkl"

python -u scripts/training/build_v24_Ilya_swa.py \
  --inputs \
    "${RUN_DIR}/checkpoints/checkpoint_step00006000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00007000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00008000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00009000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00010000.pkl" \
  --source-steps 6000 7000 8000 9000 10000 \
  --output "${RUN_DIR}/swa/swa_step06000-10000.pkl"

GPU_PEAK_MIB="$(awk -F',' 'BEGIN { peak=0 } { gsub(/ /, "", $4); if (($4 + 0) > peak) peak=$4 + 0 } END { print peak }' "${GPU_TELEMETRY}")"
CPU_MAX_RSS="$(sstat -j "${SLURM_JOB_ID}.batch" --noheader --format=MaxRSS -P 2>/dev/null | tail -n 1 || true)"
echo "resource_summary sampled_gpu_peak_mib=${GPU_PEAK_MIB} cpu_max_rss=${CPU_MAX_RSS}"
