#!/usr/bin/env bash
set -euo pipefail
RESET_PREVIEW_STAGE="${1:?stage}"
RESET_PREVIEW_CAMPAIGN="${2:?campaign}"
RESET_PREVIEW_SNAPSHOT="${3:?source snapshot}"
export PROJECT_ROOT="${PWD}"
source "${RESET_PREVIEW_SNAPSHOT}/scripts/graphcast_env.sh"
export PYTHONPATH="${RESET_PREVIEW_SNAPSHOT}:${RESET_PREVIEW_SNAPSHOT}/third_party/graphcast"
export PYTHONSAFEPATH=1 WEATHER_WORKSPACE="${PROJECT_ROOT}" WEATHER_SOURCE_ROOT="${RESET_PREVIEW_SNAPSHOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/reset-preview-${SLURM_JOB_ID:-local}"
if [[ "${RESET_PREVIEW_STAGE}" == run ]]; then
  unset JAX_PLATFORMS JAX_PLATFORM_NAME
else
  export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
fi
cd "${RESET_PREVIEW_SNAPSHOT}"
python -u scripts/experiments/run_v24_matched_reset_preview.py "${RESET_PREVIEW_STAGE}" --root "${RESET_PREVIEW_CAMPAIGN}"
