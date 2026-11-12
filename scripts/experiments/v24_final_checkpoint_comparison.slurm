#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:?stage}"
CAMPAIGN="${2:?campaign}"
SNAPSHOT="${3:?source snapshot}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
export PROJECT_ROOT="${PWD}"
source "${SNAPSHOT}/scripts/graphcast_env.sh"
export PYTHONPATH="${SNAPSHOT}:${SNAPSHOT}/third_party/graphcast"
export PYTHONSAFEPATH=1 WEATHER_WORKSPACE="${PROJECT_ROOT}" WEATHER_SOURCE_ROOT="${SNAPSHOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLCONFIGDIR="/tmp/final-comparison-${SLURM_JOB_ID:-local}"
if [[ "${STAGE}" == evaluate ]]; then
  unset JAX_PLATFORMS JAX_PLATFORM_NAME
else
  export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
fi
cd "${SNAPSHOT}"
case "${STAGE}" in
  materialize)
    python -u scripts/experiments/prepare_v24_final_checkpoint_comparison.py materialize --root "${CAMPAIGN}"
    ;;
  evaluate|report)
    python -u scripts/experiments/run_v24_raw_checkpoint_pilot.py "${STAGE}" --root "${CAMPAIGN}" --task "${TASK}"
    ;;
  *)
    echo "Unknown final-comparison stage: ${STAGE}" >&2
    exit 2
    ;;
esac
