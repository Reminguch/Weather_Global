#!/usr/bin/env bash
# CPU report after both complete production arrays; forecast fields are not loaded.
#SBATCH --account=dabanin
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
WORKFLOW="${1:?absolute workflow required}"
cd "${SLURM_SUBMIT_DIR}"
source scripts/graphcast_env.sh
export PYTHONPATH="${WORKFLOW}/source:${WORKFLOW}/source/third_party/graphcast"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/mpl-common-grid-merge-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
python "${WORKFLOW}/source/scripts/experiments/frozen_runtime.py" verify --source "${WORKFLOW}/source"
exec python -u "${WORKFLOW}/source/scripts/experiments/run_common_grid_2023.py" \
  merge --workflow "${WORKFLOW}"
