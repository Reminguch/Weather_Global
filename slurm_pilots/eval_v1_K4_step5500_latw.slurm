#!/usr/bin/env bash
# Quick eval (~30min): v1 K=4 step 5500 ckpt latw per-level, h=4 TF.
# Fills the gap so v1 K=4 phase has 2 latw points (step 5500 + 6000),
# enabling head-to-head against v3 K=4 trajectory.

#SBATCH --job-name=eval-v1-K4-5500
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_v1_K4_5500_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_v1_K4_5500_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

V1K4_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K4_7k

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-04-26_per_level_eval_latw
mkdir -p "${OUT_DIR}"

${PYTHON} ${SCRIPT} \
  --run-dir "${V1K4_DIR}" \
  --step 5500 \
  --num-segments 64 \
  --horizon 4 \
  --mode tf \
  --out "${OUT_DIR}/perlevel_K4_step5500_h4_tf.json"

echo
echo "DONE. Result file:"
ls -la "${OUT_DIR}/perlevel_K4_step5500_h4_tf.json"
