#!/usr/bin/env bash
# Apples-to-apples cycling-start comparison @ h={4, 6} for:
#   v3 K=1 step 4000 (h=6 only, h=4 already done)
#   v4 K=1 step 4500
#   v3 K=4 step 7000
#   v3 K=6 step 8500
# All under: --mode tf --warm-start-cycles 20 --segment-steps 2 --num-segments 64
# Question: does v4's K=1 stateful TBPTT generalize to longer eval horizons,
# and how does it compare to ckpts explicitly trained at K=4 / K=6?

#SBATCH --job-name=eval-v3v4-K146
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_v3v4_K146_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_v3v4_K146_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

V3K1_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_PARITY_K1_3500to6k
V4_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v4_K1_seg32_bptt8_lanes4_warmFromV3K1step4000_to20k
V3K4_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_GA_paperckpt_r1_in2_seg8_meshed_m5_h128_ds16_fullvars_K4_7500
V3K6_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_GA_K6_seg5_grouploss_LW1_step7000to9500

OUT=/home/lm8598/Weather_Global_experiments/results/2026-05-03_v3v4_K1K4K6_h4h6_cyc_warm20
mkdir -p "${OUT}"

WARM=20

run_eval () {
    local run_dir=$1; local step=$2; local horizon=$3; local out_path=$4
    echo "=========================================="
    echo "${run_dir##*/}/step${step} -> h=${horizon} cyc-start warm=${WARM}"
    echo "=========================================="
    ${PYTHON} ${SCRIPT} \
      --run-dir "${run_dir}" \
      --step ${step} \
      --num-segments 64 \
      --horizon ${horizon} \
      --segment-steps 2 \
      --warm-start-cycles ${WARM} \
      --mode tf \
      --out "${out_path}"
    echo
}

# v3 K=1 step 4000 — only h=6 missing (h=4 already in 2026-05-03_v3_K1_step4000_warm20)
run_eval "${V3K1_DIR}" 4000 6 "${OUT}/v3K1_step4000_h6_tf_warm${WARM}_seg2.json"

# v4 K=1 step 4500 @ h=4, h=6
run_eval "${V4_DIR}" 4500 4 "${OUT}/v4K1_step4500_h4_tf_warm${WARM}_seg2.json"
run_eval "${V4_DIR}" 4500 6 "${OUT}/v4K1_step4500_h6_tf_warm${WARM}_seg2.json"

# v3 K=4 step 7000 @ h=4, h=6
run_eval "${V3K4_DIR}" 7000 4 "${OUT}/v3K4_step7000_h4_tf_warm${WARM}_seg2.json"
run_eval "${V3K4_DIR}" 7000 6 "${OUT}/v3K4_step7000_h6_tf_warm${WARM}_seg2.json"

# v3 K=6 step 8500 @ h=4, h=6
run_eval "${V3K6_DIR}" 8500 4 "${OUT}/v3K6_step8500_h4_tf_warm${WARM}_seg2.json"
run_eval "${V3K6_DIR}" 8500 6 "${OUT}/v3K6_step8500_h6_tf_warm${WARM}_seg2.json"

echo "ALL DONE"
ls -la "${OUT}/"
