#!/usr/bin/env bash
# Apples-to-apples eval: K=4 ckpt and K=6 ckpt at the SAME horizon, so the
# K=4-vs-K=6 comparison is not confounded by eval-horizon shift.
#
# Existing data:
#   K=4 step 7000 @ horizon=4 -> Z latw Δ% = +1.65% (mean over leads +6h..+24h)
#   K=6 step 8500 @ horizon=6 -> Z latw Δ% = +2.09% (mean over leads +6h..+36h)
#
# Confound: horizon=6 averages over longer leads where baseline RMSE is bigger
# (32->48->59 m²/s² across K=2/4/6 baselines purely from horizon shift), so
# part of the +0.44pp Z jump from K=4 to K=6 may be horizon artifact.
#
# This slurm fills the missing two cells:
#   K=4 step 7000 @ horizon=6 (let K=4 ckpt roll 6 AR steps; trained for 4
#                              but rollout_ar handles arbitrary T)
#   K=6 step 8500 @ horizon=4 (let K=6 ckpt roll only 4 AR steps; sees fewer
#                              long leads so loses any horizon-artifact)
#
# Decision rule on the resulting 2x2:
#   row-wise (within ckpt, comparing horizons): isolates the horizon-artifact
#   col-wise (within horizon, comparing ckpts):  isolates real learning
#
# Each eval is ~25 min (full 64 segments, TF mode).

#SBATCH --job-name=eval-A2A-K4K6
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=02:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_a2a_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_a2a_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

K4_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_GA_paperckpt_r1_in2_seg8_meshed_m5_h128_ds16_fullvars_K4_7500
K6_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_GA_K6_seg5_grouploss_LW1_step7000to9500
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-05-01_apples_to_apples_K4_K6
mkdir -p "${OUT_DIR}"

run_eval () {
    local run_dir=$1
    local step=$2
    local horizon=$3
    local out_name=$4
    echo "=========================================="
    echo "Eval: ${run_dir##*/} step=${step} horizon=${horizon}"
    echo "=========================================="
    ${PYTHON} ${SCRIPT} \
      --run-dir "${run_dir}" \
      --step ${step} \
      --num-segments 64 \
      --horizon ${horizon} \
      --mode tf \
      --out "${OUT_DIR}/${out_name}.json"
    echo
}

# Existing measurements (already in eval_log) — but re-run for consistency
# of the eval segments used (eval_per_level.py uses a deterministic segment
# order so the four cells share the same segment pool).
run_eval "${K4_DIR}" 7000 4 "K4_step7000_h4"
run_eval "${K4_DIR}" 7000 6 "K4_step7000_h6"
run_eval "${K6_DIR}" 8500 4 "K6_step8500_h4"
run_eval "${K6_DIR}" 8500 6 "K6_step8500_h6"

echo "ALL DONE"
ls -la "${OUT_DIR}/"
