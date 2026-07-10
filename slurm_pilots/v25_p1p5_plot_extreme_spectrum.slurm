#!/usr/bin/env bash
# Generate extreme records (3-way) + spectrum (3-way) plots for v25 phase 1.5 step 5000.
# Depends on save job (10396951) completion to produce the nc file.
#SBATCH --job-name=v25-p1p5-plots
#SBATCH --account=biao
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v25_p1p5_plots_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v25_p1p5_plots_%j.err
set -euo pipefail
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python

NC_FILE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/2026-6-27-v25-extreme/v25_phase1p5_step5000_res2_extreme_records_K40_2022.nc
[ -f "${NC_FILE}" ] || { echo "ERR: nc file not present yet: ${NC_FILE}"; exit 1; }

echo "=== 3-way extreme records plot (GC base vs v22 K=18 vs v25 phase 1.5) ==="
$PYTHON /home/lm8598/Weather_Global_experiments/scripts/analyze_models/plot_v25_p1p5_extreme_records_3way.py

echo
echo "=== Spectrum plot (3-way, includes v25 phase 1.5) ==="
$PYTHON /home/lm8598/Weather_Global_experiments/scripts/analyze_models/plot_v25_spectrum.py

echo
echo "=== Final plot inventory ==="
ls -lh /home/lm8598/Weather_Global_experiments/results/2026-06-28-v25-extreme/plots/
