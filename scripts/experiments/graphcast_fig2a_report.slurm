#!/usr/bin/env bash
# Re-read 480 native Z500 prediction/truth pairs; bounded one-forecast memory.
#SBATCH --job-name=gc025-paper-report
#SBATCH --account=dabanin
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
cd /scratch/gpfs/DABANIN/iv9432/Weather_global
source scripts/graphcast_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKFLOW="${1:?workflow required}"
exec python -u "${WORKFLOW}/eval_source/scripts/analyze_models/report_graphcast_fig2a.py" --workflow "${WORKFLOW}"
