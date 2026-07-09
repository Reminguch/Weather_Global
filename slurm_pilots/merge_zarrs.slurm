#!/usr/bin/env bash
#SBATCH --job-name=merge-zarr
#SBATCH --account=biao
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/merge_zarrs_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/merge_zarrs_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

${PYTHON} -u -m src.data.merge_zarrs \
  --inputs \
    /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2019_dedup.zarr \
    /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --out /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
