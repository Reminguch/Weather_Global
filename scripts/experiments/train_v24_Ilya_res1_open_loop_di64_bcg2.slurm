#!/usr/bin/env bash
# Budget: 13435248_6/7 completed 10k res1/di16 updates in about 26.2h,
# with batch MaxRSS about 224 GiB. Di64/BCG4 preflight 13435249_1
# measured 10.06s/update after compilation. Allow validation and headroom.
#SBATCH --job-name=v24-r1-open-d64g2
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=288G
#SBATCH --time=1-18:00:00
#SBATCH --output=logs/v24_r1_open_di64_bcg2_%j.out
#SBATCH --error=logs/v24_r1_open_di64_bcg2_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
CONFIG="configs/experiments/v24_Ilya/res1_open_loop_mamba1_di64_bcg2_all24_10k.json"
RUN_DIR="artifacts/checkpoints/v24_Ilya/res1_open_loop/mamba1_di64_bcg2_all24_lr1em4_b1p9_b2p98_cos10k_seed22"
[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 1; }
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse existing run: ${RUN_DIR}" >&2; exit 1; }

export MPLCONFIGDIR="/tmp/mpl-v24-r1-open-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "job=${SLURM_JOB_ID} config=${CONFIG} run_dir=${RUN_DIR}"
echo "git_commit=$(git rev-parse HEAD)"
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
monitor_gpu > "logs/v24_r1_open_di64_bcg2_${SLURM_JOB_ID}_gpu.csv" &
GPU_MONITOR_PID=$!
cleanup() {
  kill "${GPU_MONITOR_PID}" 2>/dev/null || true
  wait "${GPU_MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

python -u scripts/training/train_v24_Ilya.py --config "${CONFIG}"
[[ -f "${RUN_DIR}/checkpoints/checkpoint_step00010000.pkl" ]]
