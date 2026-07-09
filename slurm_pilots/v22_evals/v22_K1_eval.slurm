#!/usr/bin/env bash
#SBATCH --job-name=v22-K1-eval
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22_K1_eval_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22_K1_eval_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments
$PYTHON -u /home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v20/eval_v20_rollout.py \
  --ckpt /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22/K1_from_v15v2_20k/v13_residual_step23000.pkl \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 20 --n-samples 32 \
  --out-json /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0522_v22_eval/v22_K1_K20.json
