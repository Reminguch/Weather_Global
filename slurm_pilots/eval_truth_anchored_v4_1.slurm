#!/usr/bin/env bash
# Truth-anchored AR eval (Ilya-style) for v4.1 ckpts:
#   * v4.1 warm-start step 5000  (= 1000 K=4cl steps after v3 K=1 warm)
#   * v4.1 from-0 step 500       (= 500 K=4cl steps from random init)
# Compare against v3 K=1 step 4000 (-0.18%) and v4 K=1 step 4500 (-0.20%).

#SBATCH --job-name=eval-ta-v41
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=02:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_ta_v41_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_ta_v41_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_truth_anchored.py
cd /home/lm8598/Weather_Global_experiments

V41_WARM=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v4_1_K1_K4cl_seg32_bptt8_lanes4_warmFromV3K1step4000_to10k
V41_FROM0=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v4_1_K1_K4cl_seg32_bptt8_lanes4_from0_to10k

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

run_one "${V41_WARM}"  5000 "v41_warm_step5000_truth_anchored_W${W}S${S}L${L}"
run_one "${V41_FROM0}" 500  "v41_from0_step500_truth_anchored_W${W}S${S}L${L}"

echo "ALL DONE"
ls -la "${OUT}/"
