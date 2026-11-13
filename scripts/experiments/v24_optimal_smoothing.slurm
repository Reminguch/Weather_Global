#!/usr/bin/env bash
set -euo pipefail
OPT_STAGE="${1:?stage}"
OPT_CAMPAIGN="${2:?campaign}"
OPT_SNAPSHOT="${3:?source snapshot}"
OPT_TASK="${SLURM_ARRAY_TASK_ID:-${4:-0}}"
export PROJECT_ROOT="${PWD}"
source "${OPT_SNAPSHOT}/scripts/graphcast_env.sh"
export PYTHONPATH="${OPT_SNAPSHOT}:${OPT_SNAPSHOT}/third_party/graphcast"
export PYTHONSAFEPATH=1 WEATHER_WORKSPACE="${PROJECT_ROOT}" WEATHER_SOURCE_ROOT="${OPT_SNAPSHOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLCONFIGDIR="/tmp/optimal-smoothing-${SLURM_JOB_ID:-local}"
if [[ "${OPT_STAGE}" == calibrate || "${OPT_STAGE}" == evaluate ]]; then
  unset JAX_PLATFORMS JAX_PLATFORM_NAME
else
  export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
fi
cd "${OPT_SNAPSHOT}"
python -u scripts/experiments/run_v24_optimal_smoothing.py "${OPT_STAGE}" --root "${OPT_CAMPAIGN}" --task "${OPT_TASK}"
