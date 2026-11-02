#!/usr/bin/env bash
# Render the small validated summary only; no forecast re-reading or GPU work.
#SBATCH --job-name=gc025-paper-plot
#SBATCH --account=dabanin
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
set -euo pipefail
cd /scratch/gpfs/DABANIN/iv9432/Weather_global
source scripts/graphcast_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKFLOW="${1:?workflow required}"
exec python -u "${WORKFLOW}/plot_source/scripts/analyze_models/plot_graphcast_fig2a.py" --workflow "${WORKFLOW}"
