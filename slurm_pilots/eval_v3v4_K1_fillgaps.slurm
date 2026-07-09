#!/usr/bin/env bash
# Fill missing horizons after partial completions:
#   v3 K=1 step 4000 cycling-start: h=6 (timeout)
#   v4 K=1 step 4500 cycling-start: h=4, h=6 (sigterm)
# All match the prior config: --warm-start-cycles 20 --segment-steps 2 --mode tf
# Safer slurm wallclock so 3 horizons fit comfortably.

#SBATCH --job-name=eval-v3v4-fill
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=03:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_v3v4_fill_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_v3v4_fill_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

V3_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_PARITY_K1_3500to6k
V4_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v4_K1_seg32_bptt8_lanes4_warmFromV3K1step4000_to20k
V3_OUT=/home/lm8598/Weather_Global_experiments/results/2026-05-03_v3_K1_step4000_warm20
V4_OUT=/home/lm8598/Weather_Global_experiments/results/2026-05-03_v4_K1_step4500_multi_horizon

WARM=20

run_eval () {
    local run_dir=$1
    local step=$2
    local horizon=$3
    local out_path=$4
    echo "=========================================="
    echo "ckpt=${run_dir##*/}/step${step} -> h=${horizon} cycling-start warm=${WARM}"
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

# v3 K=1 step 4000 h=6 (gap from 7632887 timeout)
run_eval "${V3_DIR}" 4000 6 "${V3_OUT}/v3K1_step4000_h6_tf_warm${WARM}_seg2.json"

# v4 K=1 step 4500 h=4 + h=6 (gap from 7636197 sigterm)
run_eval "${V4_DIR}" 4500 4 "${V4_OUT}/v4K1_step4500_h4_tf_warm${WARM}_seg2.json"
run_eval "${V4_DIR}" 4500 6 "${V4_OUT}/v4K1_step4500_h6_tf_warm${WARM}_seg2.json"

echo "ALL DONE"
ls -la "${V3_OUT}/" "${V4_OUT}/"
