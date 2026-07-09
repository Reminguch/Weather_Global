#!/usr/bin/env bash
# Smoke test for scripts/eval_truth_anchored.py (Ilya-style truth-anchored AR).
# Tiny config — verify wiring + measure per-step baseline forward time.
# Target: v3 K=6 step 8500 with W=2 S=2 L=4 B=2 num_chunks=2.

#SBATCH --job-name=eval-ta-smoke
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_ta_smoke_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_ta_smoke_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_truth_anchored.py
cd /home/lm8598/Weather_Global_experiments

V3K6_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_GA_K6_seg5_grouploss_LW1_step7000to9500
OUT=/home/lm8598/Weather_Global_experiments/results/2026-05-03_truth_anchored_smoke
mkdir -p "${OUT}"

${PYTHON} -u ${SCRIPT} \
  --run-dir "${V3K6_DIR}" \
  --step 8500 \
  --warmup-steps 2 \
  --trunk-steps 2 \
  --max-lead 4 \
  --window-batch-size 2 \
  --num-chunks 2 \
  --out "${OUT}/v3K6_smoke.json"

echo "SMOKE DONE"
ls -la "${OUT}/"
