#!/usr/bin/env bash
# Summarize the ten matched SWA(2k--7k) evaluations after the GPU array finishes.
#SBATCH --job-name=v24-r1-swa27s
#SBATCH --account=dabanin
#SBATCH --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/v24_r1_swa27_summary_%j.out
#SBATCH --error=logs/v24_r1_swa27_summary_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs results/v24/res1_optimizer_swa2k7
source scripts/graphcast_env.sh

python -u scripts/analyze_models/summarize_v24_Ilya_res1_optimizer_swa2k7.py \
  --experiment-root artifacts/checkpoints/v24_Ilya/res1_optimizer_matrix_20260904 \
  --reference-json artifacts/checkpoints/v22_final/res1_dm_7yr_k20_di_bcg_20k_20260813/di16_bcg1_closed_sg_stateful_20k/eval/cold_full_zero_exact/swa_step02000-08000.json \
  --output-csv results/v24/res1_optimizer_swa2k7/day5_day10_comparison.csv \
  --output-image results/v24/res1_optimizer_swa2k7/day5_day10_comparison.png
