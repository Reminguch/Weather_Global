#!/usr/bin/env bash
#SBATCH --job-name=v22-K14-warm-bp
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22_K14_warm_bp_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22_K14_warm_bp_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments
$PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22/K14_from_v22K1_23k/v13_residual_step26000.pkl \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 40 --warmup-steps 24 --n-samples 32 \
  --eval-mode warm_bp \
  --out-json /scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/results/2026-06-01-v22clean/eval_jsons/v22_K14_warm_bp.json
