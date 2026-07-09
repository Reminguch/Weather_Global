#!/usr/bin/env bash
# Pre-compute 2015-2021 record climatology (T2m max/min, 10m wind max per DOY)
#SBATCH --job-name=v18-records-clim
#SBATCH --account=biao
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v18_records_clim_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v18_records_clim_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/results/2026-05-18_v18_baseline_stream/analysis/compute_records_climatology.py
cd /home/lm8598/Weather_Global_experiments

${PYTHON} -u ${SCRIPT}
