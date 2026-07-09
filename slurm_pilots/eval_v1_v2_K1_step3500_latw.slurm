#!/usr/bin/env bash
# Run per-level lat-weighted eval on BOTH v1 K=1 step 3500 and v2-GA K=1 step
# 3500 ckpts. Outputs two JSON files for direct head-to-head comparison.
#
# Eval config: K=1, horizon=1, mode=tf (matches paper-comparable setup).
# eval_per_level.py was patched to read use_specialist_heads from run_config
# so v2 ckpt loads with the correct split-head param tree.

#SBATCH --job-name=eval-K1-3500-latw
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_K1_3500_latw_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_K1_3500_latw_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

V1_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_4k_cont
V2_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v2_GA_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K1_4k

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-04-29_v1_vs_v2_K1_step3500_latw
mkdir -p "${OUT_DIR}"

echo "=========================================================="
echo "v1 K=1 step 3500 lat-weighted per-level eval"
echo "=========================================================="
${PYTHON} ${SCRIPT} \
  --run-dir "${V1_DIR}" \
  --step 3500 \
  --num-segments 64 \
  --horizon 1 \
  --mode tf \
  --out "${OUT_DIR}/perlevel_v1_K1_step3500_h1_tf.json"

echo
echo "=========================================================="
echo "v2-GA K=1 step 3500 lat-weighted per-level eval"
echo "=========================================================="
${PYTHON} ${SCRIPT} \
  --run-dir "${V2_DIR}" \
  --step 3500 \
  --num-segments 64 \
  --horizon 1 \
  --mode tf \
  --out "${OUT_DIR}/perlevel_v2GA_K1_step3500_h1_tf.json"

echo
echo "DONE"
ls -la "${OUT_DIR}/"
