#!/usr/bin/env bash
#SBATCH --job-name=ngcm-mamba-smoke
#SBATCH --account=mengdiw
#SBATCH --partition=ailab
#SBATCH --qos=gpu-test
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global_rebuild/logs/gpu-smoke-%j.log
set -euo pipefail
cd /scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global_rebuild
export TMPDIR=$PWD/.tmp XDG_CACHE_HOME=$PWD/.cache MPLCONFIGDIR=$PWD/.cache/matplotlib
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH
exec ../Weather_Global/.venv_ngcm_data/bin/python -u experimental/residualmamba_NGCM/tests/gpu_smoke.py --gc-checkpoint data/prerequisites/graphcast_small.npz --output "logs/gpu-smoke-${SLURM_JOB_ID}.json"
