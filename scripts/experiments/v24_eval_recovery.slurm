#!/usr/bin/env bash
# Evaluation resources and dependencies are recorded in recovery/submissions.json.
set -euo pipefail
RECOVERY="${1:?absolute recovery directory}"
PHASE="${2:?phase}"
WORKSPACE="${3:?original workspace}"
RUNTIME="${RECOVERY}/source"
export PROJECT_ROOT="${WORKSPACE}"
source "${RUNTIME}/scripts/graphcast_env.sh"
export PYTHONPATH="${RUNTIME}:${RUNTIME}/third_party/graphcast"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
export MPLCONFIGDIR="/tmp/v24-recovery-${SLURM_JOB_ID}"
cd "${RUNTIME}"
if [[ "${PHASE}" == evaluate ]]; then
  python -u scripts/experiments/run_v24_eval_recovery.py smoke --recovery "${RECOVERY}" --task "${SLURM_ARRAY_TASK_ID:-0}"
fi
python -u scripts/experiments/run_v24_eval_recovery.py "${PHASE}" --recovery "${RECOVERY}" --task "${SLURM_ARRAY_TASK_ID:-0}"
