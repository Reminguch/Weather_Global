#!/usr/bin/env bash
# Small arrays of per-start scalar losses; no forecast fields retained during merge.
#SBATCH --job-name=v24-2023-report
#SBATCH --account=dabanin
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/v24_2023_%j.out
#SBATCH --error=logs/v24_2023_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
PHASE="${1:?phase required}"
WORKFLOW="${2:?absolute workflow path required}"
source scripts/graphcast_env.sh
export PYTHONPATH="${WORKFLOW}/source:${WORKFLOW}/source/third_party/graphcast"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/mpl-v24-2023-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
exec python -u "${WORKFLOW}/source/scripts/experiments/run_v24_2023_eval.py" "${PHASE}" --workflow "${WORKFLOW}"
