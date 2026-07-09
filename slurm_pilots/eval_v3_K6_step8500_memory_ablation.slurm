#!/usr/bin/env bash
# v3 K=6 step 8500 ablation matrix — 5 modes × 1 ckpt = 5 evals.
#
# Goal: verify the K-curriculum prediction. K=2 ablation showed 0% Mamba
# hidden contribution (T=2 is too short for memory build-up). K=4 ablation
# showed 31% hidden contribution at +24h Z (NORMAL−RESET = +0.59pp out of
# +1.88pp). K=6 with longer SSM time axis (T=6, 5 self-feed steps) should
# push that fraction higher if the curriculum hypothesis is real.
#
# Modes (per ckpt) — same matrix as K=4 ablation:
#                  hidden  | residual encoder-input | state_from_feedback path
#   1. NORMAL       on     | on                     | on
#   2. RESET        OFF    | on                     | on
#   3. ZEROPREV     on     | OFF                    | on
#   4. NOFB         on     | OFF                    | OFF
#   5. BOTHOFF      OFF    | OFF                    | OFF
#
# Reading the matrix at +36h Z (longest lead K=6 supports):
#   normal − reset    = SSM hidden contribution (the K-curriculum metric)
#   normal − zeroprev = encoder residual-input direct contribution
#   normal − nofb     = total residual-feedback contribution
#   nofb   − bothoff  = SSM hidden alone, no residual-feedback help
#   normal − bothoff  = total temporal-context lift
#
# Memory: K=6 with anchor_as_batch yields [T=6, B=S, lat, lon, F]. Eval
# uses segment_steps=5 (matches training config), so B=S=5, T*B = 30.
# T*B*M ≈ 76860 — slightly under K=4's 81984 budget, OK on gpu80.

#SBATCH --job-name=ablation-K6
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_ablation_K6_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_ablation_K6_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

V3K6_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_GA_K6_seg5_grouploss_LW1_step7000to9500

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-05-01_K6_step8500_memory_ablation
mkdir -p "${OUT_DIR}"

run_eval () {
    local step=$1
    local horizon=$2
    local mode=$3
    local extra_flags=$4
    local out_name=$5

    echo "=========================================="
    echo "Eval: K6_step${step} h=${horizon} mode=${mode}"
    echo "  extra_flags='${extra_flags}'"
    echo "=========================================="

    ${PYTHON} ${SCRIPT} \
      --run-dir "${V3K6_DIR}" \
      --step ${step} \
      --num-segments 64 \
      --horizon ${horizon} \
      --mode ${mode} \
      ${extra_flags} \
      --out "${OUT_DIR}/${out_name}.json"
    echo
}

# --------------------------------------------------------------------------
# v3 K=6 step 8500, horizon=6 TF — 5 ablation modes
# --------------------------------------------------------------------------
run_eval 8500 6 tf "" "v3K6_step8500_h6_tf_NORMAL"
run_eval 8500 6 tf "--reset-hidden" "v3K6_step8500_h6_tf_RESET"
run_eval 8500 6 tf "--zero-prev-residual" "v3K6_step8500_h6_tf_ZEROPREV"
run_eval 8500 6 tf "--zero-prev-residual --disable-state-feedback" "v3K6_step8500_h6_tf_NOFB"
run_eval 8500 6 tf "--reset-hidden --zero-prev-residual --disable-state-feedback" "v3K6_step8500_h6_tf_BOTHOFF"

echo
echo "ALL DONE"
ls -la "${OUT_DIR}/"
