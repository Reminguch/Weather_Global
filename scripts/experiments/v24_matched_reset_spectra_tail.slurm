#!/usr/bin/env bash
set -euo pipefail
RESET_TAIL_STAGE="${1:?stage}"
RESET_TAIL_CAMPAIGN="${2:?campaign}"
RESET_TAIL_SNAPSHOT="${3:?source snapshot}"
RESET_TAIL_TASK="${SLURM_ARRAY_TASK_ID:-${4:-0}}"
export PROJECT_ROOT="${PWD}"
source "${RESET_TAIL_SNAPSHOT}/scripts/graphcast_env.sh"
export PYTHONPATH="${RESET_TAIL_SNAPSHOT}:${RESET_TAIL_SNAPSHOT}/third_party/graphcast"
export PYTHONSAFEPATH=1 WEATHER_WORKSPACE="${PROJECT_ROOT}" WEATHER_SOURCE_ROOT="${RESET_TAIL_SNAPSHOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLCONFIGDIR="/tmp/reset-tail-${SLURM_JOB_ID:-local}"
if [[ "${RESET_TAIL_STAGE}" == evaluate ]]; then
  unset JAX_PLATFORMS JAX_PLATFORM_NAME
else
  export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
fi
cd "${RESET_TAIL_SNAPSHOT}"
python -u scripts/experiments/run_v24_matched_reset_spectra_tail.py "${RESET_TAIL_STAGE}" \
  --root "${RESET_TAIL_CAMPAIGN}" --task "${RESET_TAIL_TASK}"
