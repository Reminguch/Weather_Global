#!/usr/bin/env bash
# v3 K=1 step 4000 ckpt @ h=2 only, TF, seg=2, warm=16.
# Fills the h=2 hole; existing JSONs cover h={1, 4, 6, 8, 12, 24, 40}.

#SBATCH --job-name=eval-v3K1-h2
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_v3K1_h2_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_v3K1_h2_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

V3_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_PARITY_K1_3500to6k
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-05-01_v3_K1_step4000_multi_horizon

WARM=16
${PYTHON} ${SCRIPT} \
  --run-dir "${V3_DIR}" \
  --step 4000 \
  --num-segments 64 \
  --horizon 2 \
  --segment-steps 2 \
  --warm-start-cycles ${WARM} \
  --mode tf \
  --out "${OUT_DIR}/v3K1_step4000_h2_tf_warm${WARM}_seg2.json"

echo "DONE"
ls -la "${OUT_DIR}/"
