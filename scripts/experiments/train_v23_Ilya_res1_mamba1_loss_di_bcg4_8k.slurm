#!/usr/bin/env bash
# Four-way matched res1 screen:
#   sparse-vs-all-step H20 supervision x d_inner={16,32}, with BCG=4.
# The exact completed sparse di16/BCG1 sibling used 98.3 GiB host RAM and
# 21h38m for 10k updates. 192G leaves headroom for all-step truth tapes and
# di32; 36h covers the slower all-step variants.
#SBATCH --job-name=v23-r1-m1-loss-di
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/v23_r1_mamba1_loss_di_%A_%a.out
#SBATCH --error=logs/v23_r1_mamba1_loss_di_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIGS=(
  "configs/experiments/v23_Ilya/res1_mamba1_sparse_h20_di16_bcg4_8k.json"
  "configs/experiments/v23_Ilya/res1_mamba1_sparse_h20_di32_bcg4_8k.json"
  "configs/experiments/v23_Ilya/res1_mamba1_all_h20_di16_bcg4_8k.json"
  "configs/experiments/v23_Ilya/res1_mamba1_all_h20_di32_bcg4_8k.json"
)
RUN_NAMES=(
  "di16_bcg4_sparse_h20_mamba1_closed_sg_stateful_lr1em4_8k"
  "di32_bcg4_sparse_h20_mamba1_closed_sg_stateful_lr1em4_8k"
  "di16_bcg4_all_h20_mamba1_closed_sg_stateful_lr1em4_8k"
  "di32_bcg4_all_h20_mamba1_closed_sg_stateful_lr1em4_8k"
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
CONFIG="${CONFIGS[${TASK_ID}]}"
RUN_NAME="${RUN_NAMES[${TASK_ID}]}"
OUTPUT_ROOT="artifacts/checkpoints/v23_Ilya/res1_mamba1_loss_di_bcg4_8k_20260821"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
GPU_MEMORY_LOG="logs/v23_r1_mamba1_loss_di_${SLURM_ARRAY_JOB_ID}_${TASK_ID}_gpu_memory.csv"

[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 1; }
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse existing run: ${RUN_DIR}" >&2; exit 1; }

export MPLCONFIGDIR="/tmp/mpl-v23-r1-m1-loss-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "task_id=${TASK_ID}"
echo "config=${CONFIG}"
echo "run_dir=${RUN_DIR}"
echo "gpu_memory_log=${GPU_MEMORY_LOG}"
echo "git_commit=$(git rev-parse HEAD)"
git status --short
sha256sum \
  "${CONFIG}" \
  src/models/mamba/modules/temporal_mesh_mamba_Ilya.py \
  src/models/mamba/v23_Ilya/config.py \
  src/models/mamba/v23_Ilya/model.py \
  src/models/mamba/v23_Ilya/checkpoint.py \
  src/models/mamba/v23_Ilya/training/config.py \
  src/models/mamba/v23_Ilya/training/data.py \
  src/models/mamba/v23_Ilya/training/endpoint_step.py \
  src/models/mamba/v23_Ilya/training/runner.py \
  third_party/graphcast/graphcast/graphcast.py

monitor_gpu() {
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    sleep 10
  done
}

monitor_gpu > "${GPU_MEMORY_LOG}" &
GPU_MONITOR_PID=$!

cleanup() {
  kill "${GPU_MONITOR_PID}" 2>/dev/null || true
  wait "${GPU_MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python -u scripts/training/train_v23_Ilya.py --config "${CONFIG}"

build_swa() {
  local start_step="$1"
  local end_step="$2"
  local label="$3"
  local inputs=()
  local source_steps=()
  local step
  local padded
  for ((step=start_step; step<=end_step; step+=1000)); do
    printf -v padded "%08d" "${step}"
    inputs+=("${RUN_DIR}/checkpoints/checkpoint_step${padded}.pkl")
    source_steps+=("${step}")
  done
  python -u scripts/training/build_v23_Ilya_swa.py \
    --inputs "${inputs[@]}" \
    --source-steps "${source_steps[@]}" \
    --output "${RUN_DIR}/swa/${label}.pkl"
}

build_swa 2000 8000 "swa_step02000-08000"
build_swa 4000 8000 "swa_step04000-08000"

GPU_PEAK_MIB="$(awk -F',' 'BEGIN { peak=0 } { gsub(/ /, "", $4); if (($4 + 0) > peak) peak=$4 + 0 } END { print peak }' "${GPU_MEMORY_LOG}")"
CPU_MAX_RSS="$(sstat -j "${SLURM_JOB_ID}.batch" --noheader --format=MaxRSS -P 2>/dev/null | tail -n 1 || true)"
echo "resource_summary sampled_gpu_peak_mib=${GPU_PEAK_MIB} cpu_max_rss=${CPU_MAX_RSS}"
echo "Completed training and SWA: ${RUN_DIR}"
