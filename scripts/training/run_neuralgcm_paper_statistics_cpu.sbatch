#!/usr/bin/env bash
#SBATCH --job-name=ngcm-paper-stats-cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sh4809/weatherforecast/Weather_Global/logs/ngcm_aligned_20260924/%x-%j.out
#SBATCH --error=/home/sh4809/weatherforecast/Weather_Global/logs/ngcm_aligned_20260924/%x-%j.err
set -euo pipefail
ulimit -c 0
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/sh4809/weatherforecast/Weather_Global
source_root="$1"
shift
exec .venv_neuralgcm/bin/python -u "$source_root/scripts/training/fit_neuralgcm_paper_statistics.py" "$@"
