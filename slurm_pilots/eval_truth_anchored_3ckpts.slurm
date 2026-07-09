#!/usr/bin/env bash
# Truth-anchored AR eval (Ilya-style) for 3 ckpts:
#   v3 K=4 step 7000
#   v3 K=6 step 8500
#   v4 K=1 step 4500 (warm-started from v3 K=1, +500 stateful TBPTT steps)
#
# Config: W=8 (warmup truth-step), S=16 (trunk anchors), L=8 (lead 1..8 = +6h..+48h),
#         B=4 (chunks per GPU batch), num_chunks=16 (= 4 batches per ckpt).
# Sample size: 16 chunks × 16 anchors = 256 per lead bucket.
#
# Critical methodology difference vs eval_per_level.py:
#   In each branch_rollout, BASELINE (frozen GraphCast) self-feeds with
#   model_full_pred (= baseline + residual) as next-cycle input. Our existing
#   eval_per_level.py pre-computes baseline at every cycle from truth-fed
#   inputs, never AR-self-feeds baseline. This matches deployment regime.

#SBATCH --job-name=eval-ta-3ckpts
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=03:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_ta_3ckpts_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_ta_3ckpts_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_truth_anchored.py
cd /home/lm8598/Weather_Global_experiments

V3K4_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_GA_paperckpt_r1_in2_seg8_meshed_m5_h128_ds16_fullvars_K4_7500
V3K6_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_GA_K6_seg5_grouploss_LW1_step7000to9500
V4_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v4_K1_seg32_bptt8_lanes4_warmFromV3K1step4000_to20k

OUT=/home/lm8598/Weather_Global_experiments/results/2026-05-03_truth_anchored_AR_compare
mkdir -p "${OUT}"

W=8
S=16
L=8
B=4
N=16

run_one () {
    local run_dir=$1; local step=$2; local label=$3
    echo "================================================"
    echo "Truth-anchored AR: ${label}/step${step}"
    echo "  W=${W} S=${S} L=${L} B=${B} num_chunks=${N}"
    echo "================================================"
    ${PYTHON} -u ${SCRIPT} \
      --run-dir "${run_dir}" \
      --step ${step} \
      --warmup-steps ${W} \
      --trunk-steps ${S} \
      --max-lead ${L} \
      --window-batch-size ${B} \
      --num-chunks ${N} \
      --out "${OUT}/${label}.json"
    echo
}

run_one "${V3K4_DIR}" 7000 "v3K4_step7000_truth_anchored_W${W}S${S}L${L}"
run_one "${V3K6_DIR}" 8500 "v3K6_step8500_truth_anchored_W${W}S${S}L${L}"
run_one "${V4_DIR}"   4500 "v4K1_step4500_truth_anchored_W${W}S${S}L${L}"

echo "ALL DONE"
ls -la "${OUT}/"
