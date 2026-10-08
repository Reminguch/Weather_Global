#!/usr/bin/env bash
#SBATCH --job-name=ngcm-live-eval-watch
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global/logs/ngcm_weather_k20_20261003_h200_2k/live_eval/watch-%j.out
#SBATCH --error=/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global/logs/ngcm_weather_k20_20261003_h200_2k/live_eval/watch-%j.err
set -euo pipefail
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR=/scratch/gpfs/MENGDIW/sh4809/weatherforecast/tmp/live-eval-${SLURM_JOB_ID}
export XDG_CACHE_HOME=/scratch/gpfs/MENGDIW/sh4809/weatherforecast/cache
export MPLCONFIGDIR=$XDG_CACHE_HOME/matplotlib
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
cd /scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global
exec .venv_neuralgcm/bin/python -u /scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global/logs/ngcm_weather_k20_20261003_h200_2k/workflow_v1/live_eval/watch.py --root /scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global/logs/ngcm_weather_k20_20261003_h200_2k/live_eval
