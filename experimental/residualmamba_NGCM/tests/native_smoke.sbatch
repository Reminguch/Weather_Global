#!/usr/bin/env bash
#SBATCH --job-name=ngcm-native-smoke
#SBATCH --account=mengdiw
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global_rebuild/logs/native-smoke-%j.log
set -euo pipefail
cd /scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global_rebuild
export TMPDIR=$PWD/.tmp XDG_CACHE_HOME=$PWD/.cache
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 JAX_PLATFORMS=cpu
exec ../Weather_Global/.venv_ngcm_data/bin/python -u experimental/residualmamba_NGCM/tests/native_smoke.py --download-root ../Weather_Global/data/neuralgcm/prepared/era5_2015_2023_restore_20261104 --checkpoint ../Weather_Global/data/neuralgcm/checkpoints/deterministic_2_8_deg.pkl --output "logs/native-smoke-${SLURM_JOB_ID}"
