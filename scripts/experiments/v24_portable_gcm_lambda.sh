#!/usr/bin/env bash
# Activate the package's Python environment before invoking or submitting this wrapper.
set -euo pipefail
STAGE="${1:?train or finish}"
EXPERIMENT="${2:?absolute campaign path}"
PART="${3:-0}"
TASK="${4:-${SLURM_ARRAY_TASK_ID:-0}}"
SNAPSHOT="${EXPERIMENT}/source"
if [[ -n "${WEATHER_ACTIVATE_SCRIPT:-}" ]]; then
  source "${WEATHER_ACTIVATE_SCRIPT}"
fi
export PROJECT_ROOT="${EXPERIMENT}"
export PYTHONPATH="${SNAPSHOT}:${SNAPSHOT}/third_party/graphcast"
export PYTHONSAFEPATH=1 WEATHER_WORKSPACE="${EXPERIMENT}" WEATHER_SOURCE_ROOT="${SNAPSHOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export WEATHER_PREPARED_BOUNDED_READS=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/weather-portable-${SLURM_JOB_ID:-$$}"
mkdir -p "${MPLCONFIGDIR}"
cd "${SNAPSHOT}"
if [[ "${STAGE}" == finish ]]; then
  export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
  exec python -u scripts/experiments/run_v24_portable_gcm_lambda.py finish --experiment "${EXPERIMENT}"
elif [[ "${STAGE}" == train ]]; then
  unset JAX_PLATFORMS JAX_PLATFORM_NAME
  exec python -u scripts/experiments/run_v24_portable_gcm_lambda.py train \
    --experiment "${EXPERIMENT}" --task "${TASK}" --part "${PART}"
else
  echo "Unknown stage: ${STAGE}" >&2
  exit 2
fi
