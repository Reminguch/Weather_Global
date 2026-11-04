#!/usr/bin/env bash
#SBATCH --job-name=ngcm-era5-restore
#SBATCH --account=mengdiw
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global_rebuild/logs/restore_20261104/%x-%A_%a.log
set -euo pipefail
TASK_ROOT=/scratch/gpfs/MENGDIW/sh4809/weatherforecast
export TMPDIR=$TASK_ROOT/Weather_Global_rebuild/.tmp
export XDG_CACHE_HOME=$TASK_ROOT/Weather_Global_rebuild/.cache
export PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$TASK_ROOT/Weather_Global_rebuild"
exec "$TASK_ROOT/Weather_Global/.venv_ngcm_data/bin/python" -u scripts/preprocessing/restore_ngcm_2p8.py "${1:-month}" --root "$TASK_ROOT/Weather_Global/data/neuralgcm/prepared/era5_2015_2023_restore_20261104" --checkpoint "$TASK_ROOT/Weather_Global/data/neuralgcm/checkpoints/deterministic_2_8_deg.pkl"
