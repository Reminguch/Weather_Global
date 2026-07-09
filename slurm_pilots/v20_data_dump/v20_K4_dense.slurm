#!/usr/bin/env bash
#SBATCH --job-name=v20-K4-dense
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v20_K4_dense_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v20_K4_dense_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments
$PYTHON -u /home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v20/save_dense_preds_v20.py \
  --ckpt /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v20/K4_from_v19K1_23k/v13_residual_step26000.pkl \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 6 \
  --out-nc /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0520_v20_dense/v20_K4_dense_preds_2022.nc
