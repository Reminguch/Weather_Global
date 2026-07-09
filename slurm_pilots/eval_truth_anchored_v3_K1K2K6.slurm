#!/usr/bin/env bash
# Truth-anchored AR eval (Ilya-style, deployment-honest baseline AR self-feed)
# for v3 K-curriculum ckpts: K=1 step 4000, K=2 step 5500, K=6 step 8500.
# Combined with already-done v3 K=4 step 7000 truth-anchored, gives full
# K-curriculum sweep in deployment regime.
#
# Question: does the K-curriculum collapse get worse with deeper K?
# (K=4 was -2.09% overall; expect K=6 even worse if hypothesis right)

#SBATCH --job-name=eval-ta-K146
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=03:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_ta_K146_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_ta_K146_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_truth_anchored.py
cd /home/lm8598/Weather_Global_experiments

V3K1_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_PARITY_K1_3500to6k
V3K2_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_GA_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_8k
V3K6_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_GA_K6_seg5_grouploss_LW1_step7000to9500

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

run_one "${V3K1_DIR}" 4000 "v3K1_step4000_truth_anchored_W${W}S${S}L${L}"
run_one "${V3K2_DIR}" 5500 "v3K2_step5500_truth_anchored_W${W}S${S}L${L}"
run_one "${V3K6_DIR}" 8500 "v3K6_step8500_truth_anchored_W${W}S${S}L${L}"

echo "ALL DONE"
ls -la "${OUT}/"
