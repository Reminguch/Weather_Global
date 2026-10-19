#!/usr/bin/env bash
# Completed report startup failures were logic errors; spectral/AMSE arrays are small.
#SBATCH --job-name=v24-report-recovery
#SBATCH --account=dabanin
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v24_report_recovery_%j.out
#SBATCH --error=logs/v24_report_recovery_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
RECOVERY="${1:?absolute recovery path required}"
source scripts/graphcast_env.sh
export PYTHONPATH="${RECOVERY}/source:${RECOVERY}/source/third_party/graphcast"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/mpl-report-recovery-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
exec python -u "${RECOVERY}/source/scripts/experiments/run_v24_report_recovery.py" report --recovery "${RECOVERY}"
