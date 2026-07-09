#!/usr/bin/env bash
# v3 Mamba memory ablation matrix (anchor-as-batch layout, paper-comparable
# lat-weighted metrics). Two ckpts × four modes = 8 evals.
#
# Modes (per ckpt) and what each switch controls:
#                  hidden  | residual encoder-input | state_from_feedback path
#   1. NORMAL       on     | on                     | on
#   2. RESET        OFF    | on                     | on
#   3. ZEROPREV     on     | OFF                    | on
#   4. NOFB         on     | OFF                    | OFF
#   5. BOTHOFF      OFF    | OFF                    | OFF
#
# (RESET = --reset-hidden;  ZEROPREV = --zero-prev-residual;
#  NOFB = --zero-prev-residual --disable-state-feedback;
#  BOTHOFF = all three flags.)
#
# Reading the matrix (Δ% = improvement over frozen baseline):
#   normal − bothoff   = total temporal-context contribution
#   normal − reset     = SSM hidden-state contribution given residual feedback is intact
#   normal − zeroprev  = direct prev-residual encoder-channel contribution
#                        (note: state_from_feedback still carries prev_residual,
#                        so this is NOT the full residual-feedback contribution)
#   normal − nofb      = total residual-feedback contribution (encoder + state path),
#                        with hidden state still on
#   nofb   − bothoff   = SSM hidden-state contribution WITHOUT any residual-feedback
#                        (the cleanest test of whether hidden carries memory on its own)
#
# Decisions:
#   nofb − bothoff > 0          -> hidden state carries useful memory unaided
#   normal − reset ≈ 0          -> hidden adds nothing on top of residual feedback
#   normal − zeroprev ≫ 0       -> encoder residual-input channel is doing real work
#
# CRITICAL: eval must use the v3 anchor-as-batch layout
# ([T=K, B=S, lat, lon, F]) to replicate training. The eval script auto-resolves
# this from run_config.json (anchor_as_batch + anchor_as_batch_min_k vs horizon).

#SBATCH --job-name=ablation-mamba
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=05:00:00
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
    local extra_flags=$5    # space-separated CLI flag string
    local out_name=$6

    echo "=========================================="
    echo "Eval: run=${run_dir##*/} step=${step} h=${horizon} mode=${mode}"
    echo "  extra_flags='${extra_flags}'"
    echo "=========================================="

    ${PYTHON} ${SCRIPT} \
      --run-dir "${run_dir}" \
      --step ${step} \
      --num-segments 64 \
      --horizon ${horizon} \
      --mode ${mode} \
      ${extra_flags} \
      --out "${OUT_DIR}/${out_name}.json"
    echo
}

# ---------------------------------------------------------------------------
# v3 K=4 step 6000, horizon=4 TF — 5 ablation modes (run first; K=4 is the
# main carrier of any hidden-memory signal)
# ---------------------------------------------------------------------------
run_eval "${V3K4_DIR}" 6000 4 tf "" "v3K4_step6000_h4_tf_NORMAL"
run_eval "${V3K4_DIR}" 6000 4 tf "--reset-hidden" "v3K4_step6000_h4_tf_RESET"
run_eval "${V3K4_DIR}" 6000 4 tf "--zero-prev-residual" "v3K4_step6000_h4_tf_ZEROPREV"
run_eval "${V3K4_DIR}" 6000 4 tf "--zero-prev-residual --disable-state-feedback" "v3K4_step6000_h4_tf_NOFB"
run_eval "${V3K4_DIR}" 6000 4 tf "--reset-hidden --zero-prev-residual --disable-state-feedback" "v3K4_step6000_h4_tf_BOTHOFF"

# ---------------------------------------------------------------------------
# v3 K=2 step 5500, horizon=2 TF — 5 ablation modes (T=2 short, RESET/NORMAL
# gap may be small by construction; included for completeness)
# ---------------------------------------------------------------------------
run_eval "${V3K2_DIR}" 5500 2 tf "" "v3K2_step5500_h2_tf_NORMAL"
run_eval "${V3K2_DIR}" 5500 2 tf "--reset-hidden" "v3K2_step5500_h2_tf_RESET"
run_eval "${V3K2_DIR}" 5500 2 tf "--zero-prev-residual" "v3K2_step5500_h2_tf_ZEROPREV"
run_eval "${V3K2_DIR}" 5500 2 tf "--zero-prev-residual --disable-state-feedback" "v3K2_step5500_h2_tf_NOFB"
run_eval "${V3K2_DIR}" 5500 2 tf "--reset-hidden --zero-prev-residual --disable-state-feedback" "v3K2_step5500_h2_tf_BOTHOFF"

echo
echo "ALL DONE"
ls -la "${OUT_DIR}/"
