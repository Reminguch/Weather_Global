#!/usr/bin/env bash
#SBATCH --job-name=ngcm-weather-only
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --qos=gpu-test
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global/logs/ngcm_weather_lr_control_20261003_h200_2k/%x-%j.out
#SBATCH --error=/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global/logs/ngcm_weather_lr_control_20261003_h200_2k/%x-%j.err
set -euo pipefail
ulimit -c 0
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMPDIR="/scratch/gpfs/MENGDIW/sh4809/weatherforecast/tmp/ngcm-adaptive-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="/scratch/gpfs/MENGDIW/sh4809/weatherforecast/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd /scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global
exec .venv_neuralgcm/bin/python -u "/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global/logs/ngcm_weather_lr_control_20261003_h200_2k/workflow_v1/h200_exec.py" --hardware-dir "/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global/logs/ngcm_weather_lr_control_20261003_h200_2k/hardware" -- "$@"
