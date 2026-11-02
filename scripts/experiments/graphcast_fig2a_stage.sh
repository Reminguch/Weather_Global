#!/usr/bin/env bash
set -euo pipefail
cd /scratch/gpfs/DABANIN/iv9432/Weather_global
source scripts/graphcast_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
WORKFLOW="${1:?absolute workflow path required}"
exec python -u "${WORKFLOW}/stage_source/scripts/preprocessing/stage_graphcast_fig2a.py" \
  --workflow "${WORKFLOW}" --workers 8
