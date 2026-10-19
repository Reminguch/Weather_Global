#!/usr/bin/env bash
# Resources are measured and recorded by run_v24_cached_k1_k20.py at submission.
set -euo pipefail
EXPERIMENT="${1:?absolute experiment directory}"
PHASE="${2:?phase}"
WORKSPACE="${3:?workspace}"
TASK="${4:-0}"
case "${PHASE}" in
  cache-shard|k1-preflight) TASK="${SLURM_ARRAY_TASK_ID:?array task required}" ;;
esac
BLOCK="${5:-0}"
SOURCE_ROOT="${EXPERIMENT}/source"
export PROJECT_ROOT="${WORKSPACE}"
source "${SOURCE_ROOT}/scripts/graphcast_env.sh"
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/third_party/graphcast"
export PYTHONSAFEPATH=1
export WEATHER_WORKSPACE="${WORKSPACE}"
export WEATHER_SOURCE_ROOT="${SOURCE_ROOT}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
# Producer parity and consumer training have distinct recorded numerical modes.
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
case "${PHASE}" in
  pilot|cache-shard)
    unset JAX_PLATFORMS
    export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops=true"
    ;;
  k1-preflight|k1-block|k1-worker)
    unset JAX_PLATFORMS
    if python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("k1_deterministic", False) else 1)' "${EXPERIMENT}/manifest.json"; then
      export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops=true"
    fi
    ;;
  cache-release|cache-verify|k1-release|k20-release|report)
    export JAX_PLATFORMS=cpu
    ;;
  *) unset JAX_PLATFORMS ;;
esac
export MPLCONFIGDIR="/tmp/mpl-k1k20-${SLURM_JOB_ID}"
if [[ "${JAX_PLATFORMS:-}" != cpu ]]; then
  nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits -l 5 > "${EXPERIMENT}/logs/${PHASE}_${SLURM_JOB_ID}_${TASK}_gpu.csv" &
  MONITOR_PID=$!
  trap 'kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true' EXIT
fi
cd "${SOURCE_ROOT}"
python -u scripts/experiments/frozen_runtime.py exec --source "${SOURCE_ROOT}" \
  --workspace "${WORKSPACE}" --script scripts/experiments/run_v24_cached_k1_k20.py -- \
  "${PHASE}" --experiment "${EXPERIMENT}" --task "${TASK}" --block "${BLOCK}"
