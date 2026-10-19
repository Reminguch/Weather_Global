#!/usr/bin/env bash
#SBATCH --job-name=v24-cache-release
#SBATCH --account=dabanin
#SBATCH --qos=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/v24_baseline_release_%j.out
#SBATCH --error=logs/v24_baseline_release_%j.err
set -euo pipefail
WORKSPACE="${5:?Use submit_v24_baseline.py prepare --submit to pin the workspace}"
SOURCE_ROOT="${6:?Missing source snapshot; prepare a new frozen pilot workflow}"
cd "${WORKSPACE}"
export PROJECT_ROOT="${WORKSPACE}"
source "${SOURCE_ROOT}/scripts/graphcast_env.sh"
export JAX_PLATFORMS=cpu
python -u "${SOURCE_ROOT}/scripts/experiments/frozen_runtime.py" exec \
  --source "${SOURCE_ROOT}" --workspace "${WORKSPACE}" \
  --script scripts/preprocessing/submit_v24_baseline_production.py -- \
  --config "${1:?config}" --output-root "${2:?production root}" \
  --pilot-root "${3:?pilot root}" --pilot-job-id "${4:?pilot job}"
