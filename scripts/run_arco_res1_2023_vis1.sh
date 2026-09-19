#!/bin/bash
# Internet-enabled ERA5 staging, following the existing GraphCast vis1 wrappers.
set -euo pipefail

if [[ "$(hostname -s)" != "della-vis1" && "${ALLOW_NON_VIS1:-0}" != "1" ]]; then
  echo "Run this download on Internet-enabled della-vis1." >&2
  exit 2
fi

STAGING_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STAGING_PROJECT_ROOT}"
source scripts/graphcast_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

exec nice -n 10 python -u scripts/stage_arco_era5_res1.py \
  --output data/graphcast/graphcast/dataset/arco_res1_levels13_2023_eval.zarr \
  --start-time '2022-12-31T12:00:00' \
  --end-time '2024-01-10T18:00:00' \
  --workers 8 \
  --resume \
  "$@"
