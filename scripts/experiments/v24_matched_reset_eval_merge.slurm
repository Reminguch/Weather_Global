#!/usr/bin/env bash
#SBATCH --job-name=v24-reset-report
#SBATCH --account=dabanin
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/graphcast_env.sh
WORKFLOW="${1:?workflow required}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/mpl-matched-reset-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
exec python -u "${WORKFLOW}/control/report_v24_matched_reset_2023.py" --workflow "${WORKFLOW}"
