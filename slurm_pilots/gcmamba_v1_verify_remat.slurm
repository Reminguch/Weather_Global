#!/usr/bin/env bash
# Single-step non-remat vs remat equivalence check.  Cheap (~20-30 min).
# Verifies loss, predictions, and Mamba gradients match within bf16 tolerance.
#SBATCH --job-name=gcmamba-v1-verify
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/gcmamba_v1_verify_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/gcmamba_v1_verify_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

# Same Mamba config as the rolling_ar / remat smokes
$PYTHON -u scripts/training/gcmamba_v1/verify_remat_equivalence.py \
  --temporal-hidden-size 128 --temporal-d-inner 64 --temporal-d-state 16 \
  --temporal-d-conv 4 --temporal-layers 1 \
  --mamba-noise-scale 1e-2 \
  --seed 0 \
  --remat-processor-steps --remat-mesh2grid
echo "verify-remat done"
