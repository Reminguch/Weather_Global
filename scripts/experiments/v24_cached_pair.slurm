#!/usr/bin/env bash
# Smoke: canonical initialization + separate-process parity/paired benchmarks.
# Production resources are overridden using measurements from the smoke.
#SBATCH --job-name=v24-cached-pair
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40&nomig
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
STAGE="${1:?smoke, online or cached}"
EXPERIMENT="${2:?absolute experiment directory}"
SNAPSHOT="${EXPERIMENT}/source"
WORKSPACE="${3:?absolute Weather_global workspace}"
mkdir -p "${EXPERIMENT}/logs"
export PROJECT_ROOT="${WORKSPACE}"
source "${SNAPSHOT}/scripts/graphcast_env.sh"
export PYTHONPATH="${SNAPSHOT}:${SNAPSHOT}/third_party/graphcast${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false --xla_gpu_deterministic_ops=true"
export MPLCONFIGDIR="/tmp/v24-cached-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
cd "${SNAPSHOT}"
monitor_gpu() {
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    sleep 5
  done
}
monitor_gpu > "${EXPERIMENT}/logs/${STAGE}_${SLURM_JOB_ID}_gpu.csv" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true' EXIT
ARGS=(execute --experiment-root "${EXPERIMENT}" --backend "${STAGE}")
if [[ -n "${4:-}" ]]; then ARGS+=(--stop-step "$4"); fi
if [[ "${5:-0}" == 1 ]]; then ARGS+=(--resume-latest); fi
python -u scripts/experiments/run_v24_cached_pair.py "${ARGS[@]}"
