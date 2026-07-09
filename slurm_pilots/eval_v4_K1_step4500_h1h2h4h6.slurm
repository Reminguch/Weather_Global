#!/usr/bin/env bash
# v4 K=1 stateful TBPTT step 4500 ckpt @ h={1, 2, 4, 6}, TF, seg=2, warm=16.
# Companion: v3 K=1 step 4000 already evaluated at h={1, 4, 6} under
# results/2026-05-01_v3_K1_step4000_multi_horizon. We add h=2 separately.
#
# v4 step 4500 = 500 v4 steps after warm-start from v3 step 4000.
# Apples-to-apples comparison: same architecture, same frozen GraphCast,
# same warm-start; only difference is v4's K=1 stateful TBPTT.

#SBATCH --job-name=eval-v4K1-h1246
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=02:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_v4K1_h1246_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_v4K1_h1246_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

V4_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v4_K1_seg32_bptt8_lanes4_warmFromV3K1step4000_to20k
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-05-03_v4_K1_step4500_multi_horizon
mkdir -p "${OUT_DIR}"

run_eval () {
    local horizon=$1
    local seg=$2
    local warm=$3
    local out_name=$4
    echo "=========================================="
    echo "v4 K=1 step 4500 -> eval horizon=${horizon}, seg=${seg}, warm=${warm}, TF"
    echo "=========================================="
    ${PYTHON} ${SCRIPT} \
      --run-dir "${V4_DIR}" \
      --step 4500 \
      --num-segments 64 \
      --horizon ${horizon} \
      --segment-steps ${seg} \
      --warm-start-cycles ${warm} \
      --mode tf \
      --out "${OUT_DIR}/${out_name}.json"
    echo
}

WARM=20
run_eval 1 2 ${WARM} "v4K1_step4500_h1_tf_warm${WARM}_seg2"
run_eval 2 2 ${WARM} "v4K1_step4500_h2_tf_warm${WARM}_seg2"
run_eval 4 2 ${WARM} "v4K1_step4500_h4_tf_warm${WARM}_seg2"
run_eval 6 2 ${WARM} "v4K1_step4500_h6_tf_warm${WARM}_seg2"

echo "ALL DONE"
ls -la "${OUT_DIR}/"
