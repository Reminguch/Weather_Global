#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:?stage}"
EXPERIMENT="${2:?campaign path}"
SNAPSHOT="${EXPERIMENT}/source"
export PROJECT_ROOT="${PWD}"
source "${SNAPSHOT}/scripts/graphcast_env.sh"
export PYTHONPATH="${SNAPSHOT}:${SNAPSHOT}/third_party/graphcast"
export PYTHONSAFEPATH=1 WEATHER_WORKSPACE="${PROJECT_ROOT}" WEATHER_SOURCE_ROOT="${SNAPSHOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export WEATHER_PREPARED_BOUNDED_READS=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/full-gc-ar-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
case "${STAGE}" in
  swa|report) export JAX_PLATFORMS=cpu ;;
  *) unset JAX_PLATFORMS JAX_PLATFORM_NAME ;;
esac
cd "${SNAPSHOT}"
python -u scripts/experiments/run_v24_full_gc_ar.py "${STAGE}" --experiment "${EXPERIMENT}"
