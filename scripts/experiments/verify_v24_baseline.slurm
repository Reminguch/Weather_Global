#!/usr/bin/env bash
#SBATCH --job-name=v24-cache-verify
#SBATCH --account=dabanin
#SBATCH --qos=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v24_baseline_verify_%j.out
#SBATCH --error=logs/v24_baseline_verify_%j.err
set -euo pipefail
WORKSPACE="${3:?frozen workflow workspace}"
SOURCE_ROOT="${4:?Missing source snapshot; prepare a new frozen pilot workflow}"
PILOT_ROOT="${5:?original frozen pilot root}"
cd "${WORKSPACE}"
export PROJECT_ROOT="${WORKSPACE}"
source "${SOURCE_ROOT}/scripts/graphcast_env.sh"
export JAX_PLATFORMS=cpu
python -u "${SOURCE_ROOT}/scripts/experiments/frozen_runtime.py" exec \
  --source "${SOURCE_ROOT}" --workspace "${WORKSPACE}" \
  --script scripts/preprocessing/submit_v24_baseline.py -- run \
  --phase verify --pilot-root "${PILOT_ROOT}" \
  --config "${1:?config}" --output-root "${2:?cache root}"
