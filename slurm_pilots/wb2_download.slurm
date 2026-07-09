#!/usr/bin/env bash
# Download paper-standard WB2 6h ERA5 (1.0deg, 13 levels, GraphCast vars)
# for 2015-2022 from gs://weatherbench2/datasets/era5/...
# Produces wb2_res1_levels13_2015_2022.zarr matching local convention.

#SBATCH --job-name=wb2-dl
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=gpu80
#SBATCH --time=05:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/wb2_dl_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/wb2_dl_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr

${PYTHON} -u -m src.data.download_wb2_6h \
  --start 2015-01-01 --end 2022-12-31 \
  --out "${OUT}"

echo
echo "[wb2-dl] inspecting output..."
${PYTHON} -c "
import xarray as xr
ds = xr.open_zarr('${OUT}', consolidated=True)
print('dims:', dict(ds.sizes))
print('time:', str(ds.time.values[0])[:13], '..', str(ds.time.values[-1])[:13])
print('vars:', sorted(ds.data_vars))
print('lat:', ds.lat.values[[0,-1]], 'lon:', ds.lon.values[[0,-1]])
print('level:', ds.level.values.tolist() if 'level' in ds.coords else 'NONE')
print('total size:', ds.nbytes / 1e9, 'GB')
"
