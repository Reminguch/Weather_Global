#!/usr/bin/env bash
# Per-level eval at horizon=2 (6h + 12h, K=2's training horizon) for the two
# best ckpts:
#   * K=1 best: 7318972 (FullMamba K=1) step 4000
#   * K=2 best: 7334399 (FullMamba K=2) step 5000
# Baseline = frozen GraphCast_small AR-rolled to 12h (same for both ckpts).
# We average MAE over T = num_anchors * 2 positions (alternating 6h, 12h),
# matching the K=2 training loss layout.
#
# Two modes per ckpt:
#   tf : anchors teacher-forced, intra-sample self-fed (training-loss faithful)
#   ar : pure autoregressive self-feed (deployment honest, paper-comparable)
#
# This answers: in K=2's own arena (6h+12h), how many of 83 channels does each
# model improve, and does K=2 actually beat K=1 once we test on the horizon
# K=2 was trained for?

#SBATCH --job-name=mz-perlvl-h2
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_perlvl_h2_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_perlvl_h2_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-04-25_per_level_eval
mkdir -p "${OUT_DIR}"

K1_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_4k_cont
K2_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_5k

# K=1 best, horizon=2, AR (paper-comparable)
${PYTHON} ${SCRIPT} \
  --run-dir "${K1_DIR}" --step 4000 \
  --num-segments 64 --horizon 2 --mode ar \
  --out "${OUT_DIR}/perlevel_K1_FullMamba_step4000_h2_ar.json"

# K=1 best, horizon=2, TF (training-loss faithful)
${PYTHON} ${SCRIPT} \
  --run-dir "${K1_DIR}" --step 4000 \
  --num-segments 64 --horizon 2 --mode tf \
  --out "${OUT_DIR}/perlevel_K1_FullMamba_step4000_h2_tf.json"

# K=2 best, horizon=2, AR
${PYTHON} ${SCRIPT} \
  --run-dir "${K2_DIR}" --step 5000 \
  --num-segments 64 --horizon 2 --mode ar \
  --out "${OUT_DIR}/perlevel_K2_FullMamba_step5000_h2_ar.json"

# K=2 best, horizon=2, TF
${PYTHON} ${SCRIPT} \
  --run-dir "${K2_DIR}" --step 5000 \
  --num-segments 64 --horizon 2 --mode tf \
  --out "${OUT_DIR}/perlevel_K2_FullMamba_step5000_h2_tf.json"
