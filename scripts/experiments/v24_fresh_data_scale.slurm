#!/usr/bin/env bash
# Resources and dependencies are supplied by run_v24_fresh_data_scale.py.
set -euo pipefail
CAMPAIGN="${1:?campaign root}"
STAGE="${2:?stage key}"
WORKSPACE="${3:?workspace}"
SOURCE="${CAMPAIGN}/source"
export PROJECT_ROOT="${WORKSPACE}"
source "${SOURCE}/scripts/graphcast_env.sh"
export WEATHER_PREPARED_BOUNDED_READS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1
export TF_DETERMINISTIC_OPS=0 TF_CUDNN_DETERMINISTIC=0
if [[ -n "${SLURM_JOB_GPUS:-}" || -n "${SLURM_STEP_GPUS:-}" ]]; then
  unset JAX_PLATFORMS JAX_PLATFORM_NAME
else
  export JAX_PLATFORMS=cpu
fi
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
if [[ "${STAGE}" == cache_generate || "${STAGE}" == aligned_prepare ]]; then
  export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops=true"
fi
export MPLCONFIGDIR="/tmp/v24-fresh-scale-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
ARGS=(execute --campaign-root "${CAMPAIGN}" --stage "${STAGE}")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then ARGS+=(--index "${SLURM_ARRAY_TASK_ID}"); fi
python -u "${SOURCE}/scripts/experiments/frozen_runtime.py" exec \
  --source "${SOURCE}" --workspace "${WORKSPACE}" \
  --script scripts/experiments/run_v24_fresh_data_scale.py -- "${ARGS[@]}"
