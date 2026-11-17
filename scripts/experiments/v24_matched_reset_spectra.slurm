#!/usr/bin/env bash
set -euo pipefail
RESET_SPECTRA_STAGE="${1:?stage}"
RESET_SPECTRA_CAMPAIGN="${2:?campaign}"
RESET_SPECTRA_SNAPSHOT="${3:?source snapshot}"
RESET_SPECTRA_TASK="${SLURM_ARRAY_TASK_ID:-${4:-0}}"
export PROJECT_ROOT="${PWD}"
source "${RESET_SPECTRA_SNAPSHOT}/scripts/graphcast_env.sh"
export PYTHONPATH="${RESET_SPECTRA_SNAPSHOT}:${RESET_SPECTRA_SNAPSHOT}/third_party/graphcast"
export PYTHONSAFEPATH=1 WEATHER_WORKSPACE="${PROJECT_ROOT}" WEATHER_SOURCE_ROOT="${RESET_SPECTRA_SNAPSHOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLCONFIGDIR="/tmp/reset-spectra-${SLURM_JOB_ID:-local}"
if [[ "${RESET_SPECTRA_STAGE}" == evaluate ]]; then
  unset JAX_PLATFORMS JAX_PLATFORM_NAME
else
  export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
fi
cd "${RESET_SPECTRA_SNAPSHOT}"
python -u scripts/experiments/run_v24_matched_reset_spectra.py "${RESET_SPECTRA_STAGE}" \
  --root "${RESET_SPECTRA_CAMPAIGN}" --task "${RESET_SPECTRA_TASK}"
