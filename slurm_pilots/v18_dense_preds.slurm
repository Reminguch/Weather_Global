#!/usr/bin/env bash
# Dense rollout for v18: ~363 anchors @ 00:00 each day of 2022, K=6 each.
# Saves baseline + v18 predictions at 5 cities + global means for plotting
# timeseries of forecasts vs truth across a full year.
#SBATCH --job-name=v18-dense
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v18_dense_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v18_dense_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v18/save_dense_preds.py
cd /home/lm8598/Weather_Global_experiments

CKPT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v18_baseline_stream/K4_from_v15v2_20k/v13_residual_step23000.pkl
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v18_baseline_stream/K4_from_v15v2_20k/dense_preds_2022.nc

${PYTHON} -u ${SCRIPT} \
  --ckpt "${CKPT}" \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 6 \
  --out-nc "${OUT}"
