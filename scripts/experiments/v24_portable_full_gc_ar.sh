#!/usr/bin/env bash
# Activate the package environment before calling this cluster-neutral wrapper.
set -euo pipefail
STAGE="${1:?prepare locally, then supply train or finish}"
EXPERIMENT="${2:?absolute experiment path}"
PART="${3:-1}"
TASK="${4:-${SLURM_ARRAY_TASK_ID:-0}}"
SNAPSHOT="${EXPERIMENT}/source"
if [[ -n "${WEATHER_ACTIVATE_SCRIPT:-}" ]]; then
  source "${WEATHER_ACTIVATE_SCRIPT}"
fi
export PYTHONPATH="${SNAPSHOT}:${SNAPSHOT}/third_party/graphcast"
export PYTHONSAFEPATH=1 WEATHER_SOURCE_ROOT="${SNAPSHOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export WEATHER_PREPARED_BOUNDED_READS=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/portable-fullgc-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
if [[ "${STAGE}" == finish ]]; then
  export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
elif [[ "${STAGE}" == train ]]; then
  unset JAX_PLATFORMS JAX_PLATFORM_NAME
else
  echo "Wrapper stage must be train or finish" >&2
  exit 2
fi
cd "${SNAPSHOT}"
python -u scripts/experiments/run_v24_portable_full_gc_ar.py "${STAGE}" \
  --experiment "${EXPERIMENT}" --task "${TASK}" --part "${PART}"
