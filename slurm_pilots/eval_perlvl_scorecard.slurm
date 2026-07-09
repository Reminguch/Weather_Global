#!/usr/bin/env bash
# Per-(variable × level × lead_time) scorecard eval, GraphCast/WB2 style.
# Re-runs K=4 step 6000 (horizon=4) and K=6 step 8000 (horizon=6) with the
# updated eval_per_level.py that now emits per_channel_per_leadtime alongside
# the existing per-channel-aggregate output.
#
# Targets per ckpt:
#   K=4: 4 lead times × 83 channels = 332 targets
#   K=6: 6 lead times × 83 channels = 498 targets
# Both lat-weighted (paper-comparable). All Δ% relative to frozen GraphCast small.

#SBATCH --job-name=mz-scorecard
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_scorecard_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_scorecard_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-04-26_scorecard
mkdir -p "${OUT_DIR}"

K4_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K4_7k
K6_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K6_8k

# K=4 step 6000 -> horizon=4 (4 lead times: +6, +12, +18, +24h)
${PYTHON} ${SCRIPT} \
  --run-dir "${K4_DIR}" --step 6000 \
  --num-segments 64 --horizon 4 --segment-steps 8 --mode tf \
  --out "${OUT_DIR}/scorecard_K4_step6000_h4_tf.json"
${PYTHON} ${SCRIPT} \
  --run-dir "${K4_DIR}" --step 6000 \
  --num-segments 64 --horizon 4 --segment-steps 8 --mode ar \
  --out "${OUT_DIR}/scorecard_K4_step6000_h4_ar.json"

# K=6 step 8000 -> horizon=6 (6 lead times: +6, +12, +18, +24, +30, +36h)
${PYTHON} ${SCRIPT} \
  --run-dir "${K6_DIR}" --step 8000 \
  --num-segments 64 --horizon 6 --segment-steps 8 --mode tf \
  --out "${OUT_DIR}/scorecard_K6_step8000_h6_tf.json"
${PYTHON} ${SCRIPT} \
  --run-dir "${K6_DIR}" --step 8000 \
  --num-segments 64 --horizon 6 --segment-steps 8 --mode ar \
  --out "${OUT_DIR}/scorecard_K6_step8000_h6_ar.json"
