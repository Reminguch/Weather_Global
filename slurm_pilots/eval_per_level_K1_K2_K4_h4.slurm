#!/usr/bin/env bash
# Per-level eval at horizon=4 (6h+12h+18h+24h, K=4's training horizon) for the
# three best ckpts:
#   * K=1 best: FullMamba K=1 step 4000
#   * K=2 best: FullMamba K=2 step 5000
#   * K=4 best: FullMamba K=4 step 6000 (just plateaued)
# Baseline = frozen GraphCast_small AR-rolled to 24h (same for all three).
# This is the apples-to-apples comparison: K=1 and K=2 are OOD on horizon=4,
# K=4 is in-distribution. We average MAE over T = 64 segments × 32 positions
# and break down per-channel (83 channels) so we can answer:
#   - Does K=4 win K=1/K=2 once tested on its own horizon?
#   - Per-channel: how many of 83 improve under each K?
#   - paper-weighted: which K wins on the headline metric?
#
# Two modes per ckpt: tf (training-loss faithful) and ar (deployment-honest).

#SBATCH --job-name=mz-perlvl-h4
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=06:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_perlvl_h4_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_perlvl_h4_%j.err

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
K4_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K4_7k

# K=1 step 4000 on horizon=4 (OOD - K=1 was trained for 6h only)
${PYTHON} ${SCRIPT} \
  --run-dir "${K1_DIR}" --step 4000 \
  --num-segments 64 --horizon 4 --segment-steps 8 --mode tf \
  --out "${OUT_DIR}/perlevel_K1_FullMamba_step4000_h4_tf.json"
${PYTHON} ${SCRIPT} \
  --run-dir "${K1_DIR}" --step 4000 \
  --num-segments 64 --horizon 4 --segment-steps 8 --mode ar \
  --out "${OUT_DIR}/perlevel_K1_FullMamba_step4000_h4_ar.json"

# K=2 step 5000 on horizon=4 (OOD - K=2 was trained for 12h)
${PYTHON} ${SCRIPT} \
  --run-dir "${K2_DIR}" --step 5000 \
  --num-segments 64 --horizon 4 --segment-steps 8 --mode tf \
  --out "${OUT_DIR}/perlevel_K2_FullMamba_step5000_h4_tf.json"
${PYTHON} ${SCRIPT} \
  --run-dir "${K2_DIR}" --step 5000 \
  --num-segments 64 --horizon 4 --segment-steps 8 --mode ar \
  --out "${OUT_DIR}/perlevel_K2_FullMamba_step5000_h4_ar.json"

# K=4 step 6000 on horizon=4 (in-distribution, paper-quality number)
${PYTHON} ${SCRIPT} \
  --run-dir "${K4_DIR}" --step 6000 \
  --num-segments 64 --horizon 4 --segment-steps 8 --mode tf \
  --out "${OUT_DIR}/perlevel_K4_FullMamba_step6000_h4_tf.json"
${PYTHON} ${SCRIPT} \
  --run-dir "${K4_DIR}" --step 6000 \
  --num-segments 64 --horizon 4 --segment-steps 8 --mode ar \
  --out "${OUT_DIR}/perlevel_K4_FullMamba_step6000_h4_ar.json"
