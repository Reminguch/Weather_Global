#!/usr/bin/env bash
# Activate the package environment before invoking or submitting this wrapper.
set -euo pipefail
STAGE="${1:?calibrate, fit, evaluate, report, sweep-evaluate, or sweep-report}"
EXPERIMENT="${2:?absolute campaign path}"
TASK="${3:-${SLURM_ARRAY_TASK_ID:-0}}"
SNAPSHOT="${EXPERIMENT}/source"
if [[ -n "${WEATHER_ACTIVATE_SCRIPT:-}" ]]; then
  source "${WEATHER_ACTIVATE_SCRIPT}"
fi
export PROJECT_ROOT="${EXPERIMENT}"
export PYTHONPATH="${SNAPSHOT}:${SNAPSHOT}/third_party/graphcast"
export PYTHONSAFEPATH=1 WEATHER_WORKSPACE="${EXPERIMENT}" WEATHER_SOURCE_ROOT="${SNAPSHOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export WEATHER_PREPARED_BOUNDED_READS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/weather-smoothing-${SLURM_JOB_ID:-$$}"
mkdir -p "${MPLCONFIGDIR}"
cd "${SNAPSHOT}"
case "${STAGE}" in
  calibrate|evaluate|sweep-evaluate)
    unset JAX_PLATFORMS JAX_PLATFORM_NAME
    exec python -u scripts/experiments/run_v24_portable_smoothing.py "${STAGE}" \
      --experiment "${EXPERIMENT}" --task "${TASK}"
    ;;
  fit|report|sweep-report)
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
    exec python -u scripts/experiments/run_v24_portable_smoothing.py "${STAGE}" --experiment "${EXPERIMENT}"
    ;;
  *) echo "Unknown smoothing stage: ${STAGE}" >&2; exit 2 ;;
esac
