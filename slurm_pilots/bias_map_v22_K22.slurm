#!/usr/bin/env bash
# Per-cell bias map at lead 240h for v22 K=22 (warm_bp eval mode).
# Computes lat-weighted mean(bias²) vs mean(variance) — verifies the
# bias-variance decomposition claim.
#SBATCH --job-name=v22-bias-map
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/bias_map_v22_K22_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/bias_map_v22_K22_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/results/2026-06-12-bias-maps
mkdir -p ${OUT_DIR}

$PYTHON -u scripts/training/full_mamba_v23/compute_per_cell_bias_map.py \
  --ckpt /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22/K22_from_v22K1_23k/v13_residual_step26000.pkl \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --n-anchors 16 --warmup-steps 24 --target-steps 40 \
  --variable 2m_temperature --lead-step-index 39 \
  --seed 0 \
  --out-nc ${OUT_DIR}/v22_K22_2mT_lead240h_bias_map.nc
echo "v22 K22 bias map done"
