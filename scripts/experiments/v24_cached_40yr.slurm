#!/usr/bin/env bash
# Resources and dependencies are supplied by run_v24_cached_40yr.py.
set -euo pipefail
CAMPAIGN="${1:?campaign root}"
STAGE="${2:?stage key}"
WORKSPACE="${3:?frozen workspace}"
SOURCE="${CAMPAIGN}/source"
export PROJECT_ROOT="${WORKSPACE}"
source "${SOURCE}/scripts/graphcast_env.sh"
export WEATHER_PREPARED_BOUNDED_READS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1
if [[ -n "${SLURM_JOB_GPUS:-}" || -n "${SLURM_STEP_GPUS:-}" ]]; then
  unset JAX_PLATFORMS JAX_PLATFORM_NAME
else
  export JAX_PLATFORMS=cpu
fi
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
if [[ "${STAGE}" == cache_* && "${STAGE}" != *_collect && "${STAGE}" != cache_release && "${STAGE}" != cache_verify ]]; then
  export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops=true"
fi
export MPLCONFIGDIR="/tmp/v24-cached40-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
ARGS=(execute --campaign-root "${CAMPAIGN}" --stage "${STAGE}")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then ARGS+=(--index "${SLURM_ARRAY_TASK_ID}"); fi
python -u "${SOURCE}/scripts/experiments/frozen_runtime.py" exec \
  --source "${SOURCE}" --workspace "${WORKSPACE}" \
  --script scripts/experiments/run_v24_cached_40yr.py -- "${ARGS[@]}"
