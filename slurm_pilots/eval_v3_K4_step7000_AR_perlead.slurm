#!/usr/bin/env bash
# v3 K=4 step 7000 ckpt @ horizon=4 + extended {6, 8, 12}, AR mode.
# Mirror of eval_v3_K4_step7000_perlead.slurm but in AR mode.
#
# AR @ h=4 keeps the same horizon as the TF perlead eval; extra h={6,8,12}
# tests how well K=4-trained memory generalizes past training horizon when
# the forecast is fully autoregressive (no anchor truth-resets).

#SBATCH --job-name=eval-K4-AR
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_K4_AR_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_K4_AR_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

V3K4_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_GA_paperckpt_r1_in2_seg8_meshed_m5_h128_ds16_fullvars_K4_7500
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-05-03_v3_K4_step7000_AR_multi_horizon
mkdir -p "${OUT_DIR}"

run_eval () {
    local horizon=$1
    local seg=$2
    local out_name=$3
    echo "=========================================="
    echo "v3 K=4 step 7000 -> AR horizon=${horizon}, seg=${seg}"
    echo "=========================================="
    ${PYTHON} ${SCRIPT} \
      --run-dir "${V3K4_DIR}" \
      --step 7000 \
      --num-segments 64 \
      --horizon ${horizon} \
      --segment-steps ${seg} \
      --mode ar \
      --out "${OUT_DIR}/${out_name}.json"
    echo
}

run_eval 4  4 "v3K4_step7000_h4_ar_seg4"
run_eval 6  3 "v3K4_step7000_h6_ar_seg3"
run_eval 8  3 "v3K4_step7000_h8_ar_seg3"
run_eval 12 2 "v3K4_step7000_h12_ar_seg2"

echo "ALL DONE"
ls -la "${OUT_DIR}/"
