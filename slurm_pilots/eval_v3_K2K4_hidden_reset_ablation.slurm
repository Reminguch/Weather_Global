#!/usr/bin/env bash
# Mamba memory ablation: hidden-state reset eval on v3 K=2 step 5500 + v3 K=4
# step 6000 ckpts. Each ckpt eval'd twice — once normal (h carries across t),
# once with --reset-hidden (h zero'd at every t).
#
# If normal Δ% >> reset Δ%, Mamba memory is real ✓
# If normal Δ% ≈ reset Δ%, Mamba is acting as per-step residual head (no real
# memory contribution); paper's "Mamba unlocks memory" claim doesn't hold ✗

#SBATCH --job-name=ablation-mamba
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_ablation_mamba_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_ablation_mamba_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

V3K2_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_GA_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_8k
V3K4_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_GA_paperckpt_r1_in2_seg8_meshed_m5_h128_ds16_fullvars_K4_7500

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-04-30_mamba_memory_ablation
mkdir -p "${OUT_DIR}"

run_eval () {
    local run_dir=$1
    local step=$2
    local horizon=$3
    local mode=$4
    local reset_flag=$5     # "" or "--reset-hidden"
    local out_name=$6

    echo "=========================================="
    echo "Eval: run=${run_dir##*/} step=${step} h=${horizon} mode=${mode} reset=${reset_flag}"
    echo "=========================================="

    ${PYTHON} ${SCRIPT} \
      --run-dir "${run_dir}" \
      --step ${step} \
      --num-segments 64 \
      --horizon ${horizon} \
      --mode ${mode} \
      ${reset_flag} \
      --out "${OUT_DIR}/${out_name}.json"
    echo
}

# v3 K=2 step 5500, h=2 TF — normal vs reset
run_eval "${V3K2_DIR}" 5500 2 tf "" "v3K2_step5500_h2_tf_NORMAL"
run_eval "${V3K2_DIR}" 5500 2 tf "--reset-hidden" "v3K2_step5500_h2_tf_RESET"

# v3 K=4 step 6000, h=4 TF — normal vs reset
run_eval "${V3K4_DIR}" 6000 4 tf "" "v3K4_step6000_h4_tf_NORMAL"
run_eval "${V3K4_DIR}" 6000 4 tf "--reset-hidden" "v3K4_step6000_h4_tf_RESET"

echo
echo "ALL DONE"
ls -la "${OUT_DIR}/"
