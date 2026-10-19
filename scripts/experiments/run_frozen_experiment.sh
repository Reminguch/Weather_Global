#!/usr/bin/env bash
# Shared implementation for batch4/spatial stage launchers; sourced from snapshot.
run_frozen_experiment() {
  local workspace="$1" experiment="$2" driver="$3" stage="$4" task="$5"
  local source_root="${experiment}/source"
  if [[ ! -f "${source_root}/snapshot.json" ]]; then
    echo "No complete source snapshot; prepare a new experiment or use an explicit historical recovery: ${experiment}" >&2
    return 1
  fi
  export PROJECT_ROOT="${workspace}"
  source "${source_root}/scripts/graphcast_env.sh"
  export PYTHONPATH="${source_root}:${source_root}/third_party/graphcast"
  export PYTHONSAFEPATH=1
  export WEATHER_WORKSPACE="${workspace}"
  export WEATHER_SOURCE_ROOT="${source_root}"
  export MPLCONFIGDIR="/tmp/mpl-v24-${SLURM_JOB_ID:-local}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_ALLOCATOR=platform
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
  if [[ "${stage}" != summarize ]] && command -v nvidia-smi >/dev/null 2>&1; then
    mkdir -p "${experiment}/logs"
    local gpu_log="${experiment}/logs/${stage}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${task}_gpu.csv"
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits -l 5 > "${gpu_log}" &
    local monitor_pid=$!
    # Expand the PID now: local variables are gone when the caller exits.
    trap "kill ${monitor_pid} 2>/dev/null || true; wait ${monitor_pid} 2>/dev/null || true" EXIT
  fi
  cd "${source_root}"
  python -u scripts/experiments/frozen_runtime.py exec --source "${source_root}" \
    --workspace "${workspace}" --script "${driver}" -- \
    "${stage}" --experiment "${experiment}" --task "${task}"
}
