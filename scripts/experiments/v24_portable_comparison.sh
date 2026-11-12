#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:?evaluate or report}"
EXPERIMENT="${2:?absolute comparison directory}"
TASK="${3:-${SLURM_ARRAY_TASK_ID:-0}}"
SNAPSHOT="${EXPERIMENT}/source"
if [[ -n "${WEATHER_ACTIVATE_SCRIPT:-}" ]]; then
  source "${WEATHER_ACTIVATE_SCRIPT}"
fi
export PYTHONPATH="${SNAPSHOT}:${SNAPSHOT}/third_party/graphcast"
export PYTHONSAFEPATH=1 WEATHER_WORKSPACE="${PWD}" WEATHER_SOURCE_ROOT="${SNAPSHOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export WEATHER_PREPARED_BOUNDED_READS=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/portable-comparison-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
if [[ "${STAGE}" == report ]]; then
  export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
else
  unset JAX_PLATFORMS JAX_PLATFORM_NAME
fi
cd "${SNAPSHOT}"
python -u scripts/experiments/run_v24_portable_comparison.py "${STAGE}" --experiment "${EXPERIMENT}" --task "${TASK}"
