#!/usr/bin/env bash
# Per-level eval over the full 2022 val set for the two best ckpts:
#   * K=1 best: 7318972 (FullMamba K=1) step 4000
#   * K=2 best: 7334399 (FullMamba K=2) step 5000
# Outputs JSON with 83 (variable × level) channel breakdown of baseline vs MZ
# MAE/RMSE so we can count how many channels improved.

#SBATCH --job-name=mz-perlevel
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_perlevel_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_perlevel_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-04-25_per_level_eval
mkdir -p "${OUT_DIR}"

# K=1 best
${PYTHON} ${SCRIPT} \
  --run-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_4k_cont \
  --step 4000 \
  --num-segments 64 \
  --out "${OUT_DIR}/perlevel_K1_FullMamba_step4000.json"

# K=2 best
${PYTHON} ${SCRIPT} \
  --run-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_5k \
  --step 5000 \
  --num-segments 64 \
  --out "${OUT_DIR}/perlevel_K2_FullMamba_step5000.json"
