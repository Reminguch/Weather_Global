#!/usr/bin/env bash
# Multi-anchor NYC 2m_T trace for v22 K=22 (open-loop ckpt). 32 anchors,
# fixed-pool sampling (matches eval_v24_closed.py). Used to visualize the
# small ~2.5% RMSE improvement by averaging across anchors.
#SBATCH --job-name=nyc-multi-v22K22
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=02:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/nyc_multi_v22K22_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/nyc_multi_v22K22_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/results/2026-06-05-city-traces
mkdir -p ${OUT_DIR}

$PYTHON -u scripts/training/full_mamba_v23/eval_nyc_multi_anchor.py \
  --ckpt /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22/K22_from_v22K1_23k/v13_residual_step26000.pkl \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2 \
  --target-steps 40 \
  --n-samples 32 \
  --seed 0 \
  --out-json ${OUT_DIR}/v22_K22_nyc_multi_anchor.json
echo "v22 K=22 multi-anchor NYC trace done"
