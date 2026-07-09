#!/usr/bin/env bash
# v4 K=1 step 4500 cycling-start eval @ h={4, 6}, exact-config-matched to:
#   h=4  -> v3 K=4 step 7000 perlead   (seg=8, no warm, num_seg=64, mode=tf)
#   h=6  -> v3 K=6 step 8500 mh        (seg=5, no warm, num_seg=64, mode=tf)
# So v4 numbers are directly comparable to existing v3 K=4 / K=6 reference numbers
# without re-running v3.

#SBATCH --job-name=eval-v4-h4h6
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_v4_h4h6_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_v4_h4h6_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

V4_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v4_K1_seg32_bptt8_lanes4_warmFromV3K1step4000_to20k
OUT=/home/lm8598/Weather_Global_experiments/results/2026-05-03_v4_K1_step4500_h4h6_match_v3K4K6
mkdir -p "${OUT}"

run_eval () {
    local horizon=$1; local seg=$2; local out_name=$3
    echo "=========================================="
    echo "v4 K=1 step 4500 -> h=${horizon}, seg=${seg}, no warm, mode=tf"
    echo "=========================================="
    ${PYTHON} ${SCRIPT} \
      --run-dir "${V4_DIR}" \
      --step 4500 \
      --num-segments 64 \
      --horizon ${horizon} \
      --segment-steps ${seg} \
      --mode tf \
      --out "${OUT}/${out_name}.json"
    echo
}

# Match v3 K=4 step 7000 perlead config at h=4 (seg=8)
run_eval 4 8 "v4K1_step4500_h4_tf_seg8"

# Match v3 K=6 step 8500 multi-horizon config at h=6 (seg=5)
run_eval 6 5 "v4K1_step4500_h6_tf_seg5"

echo "ALL DONE"
ls -la "${OUT}/"
