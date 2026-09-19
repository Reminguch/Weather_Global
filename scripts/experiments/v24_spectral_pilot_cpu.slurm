#!/usr/bin/env bash
# Aggregate compact spectra and per-start losses; no field exports loaded for report.
#SBATCH --job-name=v24-spectral-report
#SBATCH --account=dabanin
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/v24_spectral_report_%j.out
#SBATCH --error=logs/v24_spectral_report_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
WORKFLOW="${1:?absolute workflow path required}"
export PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
source "${WORKFLOW}/source/scripts/graphcast_env.sh"
export PYTHONPATH="${WORKFLOW}/source:${WORKFLOW}/source/third_party/graphcast"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/mpl-v24-spectral-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
exec python -u "${WORKFLOW}/source/scripts/experiments/run_v24_spectral_pilot.py" \
  report --workflow "${WORKFLOW}"
