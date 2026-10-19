#!/usr/bin/env bash
# Resources are supplied explicitly by the recovery submission ledger.
set -euo pipefail
DIAGNOSTIC_ROOT="${1:?diagnostic directory}"
PHASE="${2:?phase}"
SOURCE_ROOT="${DIAGNOSTIC_ROOT}/source"
export PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
source "${SOURCE_ROOT}/scripts/graphcast_env.sh"
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/third_party/graphcast"
export PYTHONSAFEPATH=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/v24-numerics-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
ARGS=("${PHASE}" --root "${DIAGNOSTIC_ROOT}")
if [[ "${PHASE}" == replay ]]; then
  case "${SLURM_ARRAY_TASK_ID:?array required}" in
    0) ARGS+=(--arm mgda) ;;
    1) ARGS+=(--arm mse) ;;
    *) exit 2 ;;
  esac
fi
if [[ "${PHASE}" == summarize ]]; then
  export JAX_PLATFORMS=cpu
else
  unset JAX_PLATFORMS
fi
cd "${PROJECT_ROOT}"
python -u "${SOURCE_ROOT}/scripts/experiments/run_v24_recovery_diagnostics.py" "${ARGS[@]}"
