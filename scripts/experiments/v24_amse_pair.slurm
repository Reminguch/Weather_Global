#!/usr/bin/env bash
# Resources supplied by the gated submission driver, from observed job history.
set -euo pipefail
STAGE="${1:?stage}"
EXPERIMENT="${2:?absolute experiment path}"
TASK="${3:-${SLURM_ARRAY_TASK_ID:-0}}"
SNAPSHOT="${EXPERIMENT}/source"
export PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
source "${SNAPSHOT}/scripts/graphcast_env.sh"
export PYTHONPATH="${SNAPSHOT}:${SNAPSHOT}/third_party/graphcast${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
export MPLCONFIGDIR="/tmp/v24-amse-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
cd "${SNAPSHOT}"
if [[ "${STAGE}" == evaluate || "${STAGE}" == report ]]; then
  python -u scripts/experiments/run_v24_amse_pair.py verify --experiment "${EXPERIMENT}"
  python -u scripts/analyze_models/evaluate_v24_amse_pair.py "${STAGE}" --experiment "${EXPERIMENT}" --task "${TASK}"
else
  python -u scripts/experiments/run_v24_amse_pair.py "${STAGE}" --experiment "${EXPERIMENT}" --task "${TASK}"
fi
