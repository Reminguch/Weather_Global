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
export WEATHER_PREPARED_BOUNDED_READS=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLCONFIGDIR="/tmp/full-gc-low-lr-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
case "${STAGE}" in
  finish|validate|plan) export JAX_PLATFORMS=cpu ;;
  *)
    unset JAX_PLATFORMS JAX_PLATFORM_NAME
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits -l 5 > "${EXPERIMENT}/logs/${STAGE}_${SLURM_JOB_ID}_${TASK}_gpu.csv" &
    MONITOR_PID=$!
    trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT
    ;;
esac
cd "${SNAPSHOT}"
python -u scripts/experiments/run_v24_full_gc_low_lr_restart.py "${STAGE}" \
  --experiment "${EXPERIMENT}" --task "${TASK}" --phase "${PHASE}"
