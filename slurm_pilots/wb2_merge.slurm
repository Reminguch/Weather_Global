#!/usr/bin/env bash
# Merge 16 per-chunk WB2 zarrs into wb2_res1_levels13_2015_2022.zarr.
# CPU-only (no internet needed). merge_zarrs.py has resume — picks up
# from where the killed login-node merge left off.

#SBATCH --job-name=wb2-merge
#SBATCH --account=biao
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/wb2_merge_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/wb2_merge_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

CHUNK_DIR=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_chunks
FINAL=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr

${PYTHON} -u -m src.data.merge_zarrs \
  --inputs ${CHUNK_DIR}/wb2_2015h1.zarr ${CHUNK_DIR}/wb2_2015h2.zarr \
           ${CHUNK_DIR}/wb2_2016h1.zarr ${CHUNK_DIR}/wb2_2016h2.zarr \
           ${CHUNK_DIR}/wb2_2017h1.zarr ${CHUNK_DIR}/wb2_2017h2.zarr \
           ${CHUNK_DIR}/wb2_2018h1.zarr ${CHUNK_DIR}/wb2_2018h2.zarr \
           ${CHUNK_DIR}/wb2_2019h1.zarr ${CHUNK_DIR}/wb2_2019h2.zarr \
           ${CHUNK_DIR}/wb2_2020h1.zarr ${CHUNK_DIR}/wb2_2020h2.zarr \
           ${CHUNK_DIR}/wb2_2021h1.zarr ${CHUNK_DIR}/wb2_2021h2.zarr \
           ${CHUNK_DIR}/wb2_2022h1.zarr ${CHUNK_DIR}/wb2_2022h2.zarr \
  --out "${FINAL}"

echo "[wb2-merge] DONE"
${PYTHON} -c "
import xarray as xr
ds = xr.open_zarr('${FINAL}', consolidated=False)
print('  dims:', dict(ds.sizes))
print('  time:', str(ds.time.values[0])[:10], '->', str(ds.time.values[-1])[:10])
print('  total size:', ds.nbytes / 1e9, 'GB')
"
