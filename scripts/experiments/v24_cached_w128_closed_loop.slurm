#!/usr/bin/env bash
# Resources and dependencies are recorded by the experiment driver.
set -euo pipefail
stage="${1:?stage}"
experiment="${2:?absolute experiment directory}"
workspace="${3:?absolute workspace}"
task="${4:-${SLURM_ARRAY_TASK_ID:-0}}"
if [[ "${stage}" == summarize ]]; then
  export JAX_PLATFORMS=cpu
else
  unset JAX_PLATFORMS JAX_PLATFORM_NAME
fi
source "${experiment}/source/scripts/experiments/run_frozen_experiment.sh"
run_frozen_experiment "${workspace}" "${experiment}" \
  scripts/experiments/run_v24_cached_w128_closed_loop.py "${stage}" "${task}"
