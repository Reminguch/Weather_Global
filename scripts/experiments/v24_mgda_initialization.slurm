#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:?stage}"
EXPERIMENT="${2:?absolute campaign path}"
PHASE="${3:-0}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
SNAPSHOT="${EXPERIMENT}/source"
export PROJECT_ROOT="${PWD}"
source "${SNAPSHOT}/scripts/graphcast_env.sh"
export PYTHONPATH="${SNAPSHOT}:${SNAPSHOT}/third_party/graphcast"
export PYTHONSAFEPATH=1
export WEATHER_WORKSPACE="${PROJECT_ROOT}"
export WEATHER_SOURCE_ROOT="${SNAPSHOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
case "${STAGE}" in
  report) export JAX_PLATFORMS=cpu ;;
  *) unset JAX_PLATFORMS ;;
esac
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLCONFIGDIR="/tmp/mgda-init-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
if [[ "${STAGE}" != report ]]; then
  nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits -l 5 > "${EXPERIMENT}/logs/${STAGE}_${SLURM_JOB_ID}_${TASK}_gpu.csv" &
  MONITOR_PID=$!
  trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT
fi
cd "${SNAPSHOT}"
python -u scripts/experiments/run_v24_mgda_initialization.py "${STAGE}" \
  --experiment "${EXPERIMENT}" --task "${TASK}" --phase "${PHASE}"
