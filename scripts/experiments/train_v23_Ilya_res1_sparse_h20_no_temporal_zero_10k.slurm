#!/usr/bin/env bash
# res1 check of sparse H20 supervision with the Mamba output projections left
# at their standard initialization. The late 6k--10k checkpoints are averaged
# after training. A matched prior res1 job used 60.7 GiB host RAM and about
# 6.6 s/update; sparse targets get additional RAM and runtime headroom here.
#SBATCH --job-name=v23-r1-sp-noz
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-06:00:00
#SBATCH --output=logs/v23_r1_sparse_noz_%j.out
#SBATCH --error=logs/v23_r1_sparse_noz_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG="configs/experiments/v23_Ilya/res1_sparse_h20_di16_bcg1_no_temporal_zero_10k.json"
OUTPUT_ROOT="artifacts/checkpoints/v23_Ilya/res1_sparse_h20_no_temporal_zero_10k_20260817"
RUN_NAME="di16_bcg1_sparse_h20_closed_sg_stateful_no_tzero_lr3em6_10k"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
GPU_MEMORY_LOG="logs/v23_r1_sparse_noz_${SLURM_JOB_ID}_gpu_memory.csv"

[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 1; }
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse existing run: ${RUN_DIR}" >&2; exit 1; }

export MPLCONFIGDIR="/tmp/mpl-v23-r1-sparse-noz-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "config=${CONFIG}"
echo "run_dir=${RUN_DIR}"
echo "gpu_memory_log=${GPU_MEMORY_LOG}"
echo "git_commit=$(git rev-parse HEAD)"
git status --short
sha256sum \
  "${CONFIG}" \
  src/models/mamba/v23_Ilya/checkpoint.py \
  src/models/mamba/v23_Ilya/training/config.py \
  src/models/mamba/v23_Ilya/training/data.py \
  src/models/mamba/v23_Ilya/training/endpoint_step.py \
  src/models/mamba/v23_Ilya/training/runner.py

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

inputs=()
source_steps=()
for step in 6000 7000 8000 9000 10000; do
  inputs+=("${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl")
  source_steps+=("${step}")
done

python -u scripts/training/build_v23_Ilya_swa.py \
  --inputs "${inputs[@]}" \
  --source-steps "${source_steps[@]}" \
  --output "${RUN_DIR}/swa/swa_step06000-10000.pkl"

echo "Completed training and SWA: ${RUN_DIR}"
