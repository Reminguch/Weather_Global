#!/usr/bin/env bash
# Budgets and dependencies are supplied by the frozen campaign driver.
set -euo pipefail
STAGE="${1:?stage}"
EXPERIMENT="${2:?absolute experiment path}"
PHASE="${3:?phase}"
TASK="${4:-${SLURM_ARRAY_TASK_ID:-0}}"
SNAPSHOT="${EXPERIMENT}/source"
export PROJECT_ROOT="${PWD}"
source "${SNAPSHOT}/scripts/graphcast_env.sh"
export PYTHONPATH="${SNAPSHOT}:${SNAPSHOT}/third_party/graphcast"
export PYTHONSAFEPATH=1
export WEATHER_WORKSPACE="${PROJECT_ROOT}"
export WEATHER_SOURCE_ROOT="${SNAPSHOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLCONFIGDIR="/tmp/amse-regime-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
if [[ "${STAGE}" == smoke || "${STAGE}" == train ]]; then
  nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits -l 5 > "${EXPERIMENT}/logs/${PHASE}_${STAGE}_${SLURM_JOB_ID}_${TASK}_gpu.csv" &
  MONITOR_PID=$!
  trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT
fi
cd "${SNAPSHOT}"
python -u scripts/experiments/run_v24_amse_regime.py "${STAGE}" \
  --experiment "${EXPERIMENT}" --phase "${PHASE}" --task "${TASK}"
