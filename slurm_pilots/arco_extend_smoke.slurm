#!/usr/bin/env bash
# ARCO extend smoke: 5 days only, to validate per-day download + regrid +
# zarr write before doing 2 years.

#SBATCH --job-name=arco-smoke
#SBATCH --account=biao
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/arco_smoke_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/arco_smoke_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/arco_smoke_5d.zarr
rm -rf "${OUT}" 2>/dev/null || true

${PYTHON} -u -m src.data.extend_zarr_from_arco \
  --start 2024-01-01 --end 2024-01-05 \
  --out "${OUT}" \
  --workers 4

echo
echo "[arco-smoke] inspecting output..."
${PYTHON} -c "
import xarray as xr
ds = xr.open_zarr('${OUT}', consolidated=True)
print('dims:', dict(ds.sizes))
print('time:', str(ds.time.values[0])[:13], '..', str(ds.time.values[-1])[:13])
print('vars:', list(ds.data_vars))
print('lat:', ds.lat.values[[0,-1]], 'lon:', ds.lon.values[[0,-1]])
print('level:', ds.level.values.tolist() if 'level' in ds.coords else 'NONE')
print('total size:', ds.nbytes / 1e6, 'MB')
"
