#!/usr/bin/env bash
# Truth-anchored AR eval (Ilya-style) for v4 K=1 step 4500 only.
# v3 numbers stay as cycling-start TF (existing JSONs); v4 uses Ilya regime.
# Question: can v4 (deployment-honest baseline-AR eval) match or beat v3 K=4/K=6
# headline numbers from existing cycling-start TF eval at K>1?

#SBATCH --job-name=eval-ta-v4
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_ta_v4_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_ta_v4_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_truth_anchored.py
cd /home/lm8598/Weather_Global_experiments

V4_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v4_K1_seg32_bptt8_lanes4_warmFromV3K1step4000_to20k

OUT=/home/lm8598/Weather_Global_experiments/results/2026-05-03_truth_anchored_AR_compare
mkdir -p "${OUT}"

W=8
S=16
L=8
B=4
N=16

${PYTHON} -u ${SCRIPT} \
  --run-dir "${V4_DIR}" \
  --step 4500 \
  --warmup-steps ${W} \
  --trunk-steps ${S} \
  --max-lead ${L} \
  --window-batch-size ${B} \
  --num-chunks ${N} \
  --out "${OUT}/v4K1_step4500_truth_anchored_W${W}S${S}L${L}.json"

echo "ALL DONE"
ls -la "${OUT}/"
