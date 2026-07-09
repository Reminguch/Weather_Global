#!/usr/bin/env bash
# Re-eval K=6 best at step 8000 (just plateaued, see eval_log step 7000-8000),
# overriding the step 7500 result the main batch (job 7373823) used. The peak
# at step 8000 is TF_w +0.53%, AR_w +0.35%; step 7500 is similar but a hair
# behind, so step 8000 should be the cleanest "K=6 best" number.

#SBATCH --job-name=mz-K6-eval-h6
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_K6_h6_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_K6_h6_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-04-26_per_level_eval_latw
mkdir -p "${OUT_DIR}"

K6_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K6_8k

${PYTHON} ${SCRIPT} \
  --run-dir "${K6_DIR}" --step 8000 \
  --num-segments 64 --horizon 6 --segment-steps 8 --mode tf \
  --out "${OUT_DIR}/perlevel_K6_step8000_h6_tf.json"
${PYTHON} ${SCRIPT} \
  --run-dir "${K6_DIR}" --step 8000 \
  --num-segments 64 --horizon 6 --segment-steps 8 --mode ar \
  --out "${OUT_DIR}/perlevel_K6_step8000_h6_ar.json"
