#!/usr/bin/env bash
#SBATCH --job-name=v24-r1-baseline
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:02:00
#SBATCH --output=logs/v24_baseline_%A_%a.out
#SBATCH --error=logs/v24_baseline_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
MODE="${1:?pilot or generate}"
CONFIG="${2:?config path}"
CACHE_ROOT="${3:?cache output root}"
export MPLCONFIGDIR="/tmp/mpl-v24-baseline-${SLURM_JOB_ID}"
mkdir -p logs "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false --xla_gpu_deterministic_ops=true"
source scripts/graphcast_env.sh
# The CPU submission job exports JAX_PLATFORMS=cpu; do not inherit it here.
unset JAX_PLATFORMS JAX_PLATFORM_NAME
monitor_gpu() {
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    sleep 10
  done
}
monitor_gpu > "logs/v24_baseline_${SLURM_JOB_ID}_gpu.csv" &
MONITOR_PID=$!
cleanup() { kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; }
trap cleanup EXIT
case "${MODE}" in
  pilot) EXTRA=(--num-shards 1 --shard-index 0 --max-chunks 4 --pilot-parity) ;;
  generate) EXTRA=(--num-shards 8 --shard-index "${SLURM_ARRAY_TASK_ID:?array task required}") ;;
  *) echo "Unsupported mode ${MODE}" >&2; exit 2 ;;
esac
python -u scripts/preprocessing/precompute_v24_baseline.py --config "${CONFIG}" --output-root "${CACHE_ROOT}" --resume "${EXTRA[@]}"
