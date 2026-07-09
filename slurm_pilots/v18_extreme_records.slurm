#!/usr/bin/env bash
# Extreme-records dataset for v18: ~120 anchors @ 00:00 every 3rd day of 2022,
# K=1..6 each. Saves full global grids for 2m_T, 10m_u, 10m_v
# (truth + baseline + v18) so we can compute pointwise heat/cold/wind
# forecast bias vs record exceedance.
#SBATCH --job-name=v18-extreme
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=03:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v18_extreme_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v18_extreme_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v18/save_extreme_records_data.py
cd /home/lm8598/Weather_Global_experiments

CKPT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v18_baseline_stream/K4_from_v15v2_20k/v13_residual_step23000.pkl
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v18_baseline_stream/K4_from_v15v2_20k/extreme_records_2022.nc

${PYTHON} -u ${SCRIPT} \
  --ckpt "${CKPT}" \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 6 \
  --anchor-stride-days 3 \
  --out-nc "${OUT}"
