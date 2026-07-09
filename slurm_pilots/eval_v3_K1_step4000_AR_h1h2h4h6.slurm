#!/usr/bin/env bash
# v3 K=1 step 4000 baseline @ h={1, 2, 4, 6}, AR (autoregressive), warm=20.
# Apples-to-apples baseline for v4 step 4500 AR comparison.

#SBATCH --job-name=eval-v3K1-AR
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=02:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_v3K1_AR_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_v3K1_AR_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

V3_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_PARITY_K1_3500to6k
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-05-03_v3_K1_step4000_AR_warm20
mkdir -p "${OUT_DIR}"

run_eval () {
    local horizon=$1
    local seg=$2
    local warm=$3
    local out_name=$4
    echo "=========================================="
    echo "v3 K=1 step 4000 -> AR horizon=${horizon}, seg=${seg}, warm=${warm}"
    echo "=========================================="
    ${PYTHON} ${SCRIPT} \
      --run-dir "${V3_DIR}" \
      --step 4000 \
      --num-segments 64 \
      --horizon ${horizon} \
      --segment-steps ${seg} \
      --warm-start-cycles ${warm} \
      --mode ar \
      --out "${OUT_DIR}/${out_name}.json"
    echo
}

WARM=20
run_eval 1 2 ${WARM} "v3K1_step4000_h1_ar_warm${WARM}_seg2"
run_eval 2 2 ${WARM} "v3K1_step4000_h2_ar_warm${WARM}_seg2"
run_eval 4 2 ${WARM} "v3K1_step4000_h4_ar_warm${WARM}_seg2"
run_eval 6 2 ${WARM} "v3K1_step4000_h6_ar_warm${WARM}_seg2"

echo "ALL DONE"
ls -la "${OUT_DIR}/"
