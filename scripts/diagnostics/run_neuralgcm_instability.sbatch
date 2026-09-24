#!/usr/bin/env bash
#SBATCH --job-name=ngcm-instability-diagnostic
#SBATCH --qos=gpu-test
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100&gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/sh4809/weatherforecast/Weather_Global/logs/neuralgcm_instability_20260924/%x-%j.out
#SBATCH --error=/home/sh4809/weatherforecast/Weather_Global/logs/neuralgcm_instability_20260924/%x-%j.err
set -euo pipefail
ulimit -c 0
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/sh4809/weatherforecast/Weather_Global
exec .venv_neuralgcm/bin/python -u scripts/diagnostics/diagnose_neuralgcm_instability.py \
  --experiment-root /scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global/artifacts/checkpoints/neuralgcm_residual/res2p8_w128_256_di16_32_train2015_2021_20260920_scatter_v3 \
  --run-id "$1" --origins 4 --audit-k1 \
  --output "logs/neuralgcm_instability_20260924/${1}-${SLURM_JOB_ID}"
