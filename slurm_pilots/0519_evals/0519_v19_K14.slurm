#!/usr/bin/env bash
# 0519 v1819new (fixed AR forcings) eval, K=12 rollout (lead 6h..72h)
#SBATCH --job-name=0519_v19_K14
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/0519_v19_K14_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/0519_v19_K14_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

${PYTHON} -u /home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v18/eval_v1819new_rollout.py \
  --ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v18_v2/K14_from_K1v2_23k/v13_residual_step26000.pkl" \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 12 \
  --n-samples 32 \
  --out-json "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0519_v18_v19_eval/v19_K14.json"
