#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:?stage}"
CAMPAIGN="${2:?campaign}"
SNAPSHOT="${3:?source snapshot}"
export PROJECT_ROOT="${PWD}"
source "${SNAPSHOT}/scripts/graphcast_env.sh"
export PYTHONPATH="${SNAPSHOT}:${SNAPSHOT}/third_party/graphcast"
export PYTHONSAFEPATH=1 WEATHER_WORKSPACE="${PROJECT_ROOT}" WEATHER_SOURCE_ROOT="${SNAPSHOT}"
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLCONFIGDIR="/tmp/low-lr-annual-${SLURM_JOB_ID:-local}"
cd "${SNAPSHOT}"
python -u scripts/experiments/run_v24_low_lr_downstream.py "${STAGE}" --root "${CAMPAIGN}"
