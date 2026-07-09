#!/usr/bin/env bash
# Run on Della login node to regrid the sample dataset to 4-degree resolution.
# Usage (from project root on della.princeton.edu):
#   bash scripts/run_regrid_4deg.sh
# Or:
#   cd /scratch/gpfs/DABANIN/iv9432/Weather_global && source scripts/graphcast_env.sh && python scripts/regrid_resolution.py --input data/graphcast/graphcast/dataset/source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc --output data/graphcast/graphcast/dataset/source-era5_date-2022-01-01_res-4.0_levels-13_steps-01.nc --resolution 4

set -euo pipefail
cd /scratch/gpfs/DABANIN/iv9432/Weather_global
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/graphcast_env.sh"

# One-time if needed: pip install xarray netCDF4
python scripts/regrid_resolution.py \
  --input data/graphcast/graphcast/dataset/source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc \
  --output data/graphcast/graphcast/dataset/source-era5_date-2022-01-01_res-4.0_levels-13_steps-01.nc \
  --resolution 4
