#!/usr/bin/env bash
# v3 K=1 step 4000 ckpt evaluated at horizon ∈ {12, 24, 40}, TF mode.
# Companion to eval_v3_K1_step4000_multi_horizon.slurm (job 7564720, h=1/4/6/8)
# — together they fill h ∈ {1, 4, 6, 8, 12, 24, 40} for the K=1 ckpt, the
# same range as the K=6 step 8500 ckpt eval (slurm 7564833).
#
# Output written to the same dir as the short-horizon slurm so all K=1
# multi-horizon JSONs end up together for easy aggregation.

#SBATCH --job-name=eval-K1-longH
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=06:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_K1_longH_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_K1_longH_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

V3K1_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_PARITY_K1_3500to6k
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-05-01_v3_K1_step4000_multi_horizon
mkdir -p "${OUT_DIR}"

run_eval () {
    local horizon=$1
    local seg=$2
    local warm=$3
    local out_name=$4
    echo "=========================================="
    echo "K=1 step 4000 ckpt -> eval horizon=${horizon}, seg=${seg}, warm=${warm}, TF"
    echo "=========================================="
    ${PYTHON} ${SCRIPT} \
      --run-dir "${V3K1_DIR}" \
      --step 4000 \
      --num-segments 64 \
      --horizon ${horizon} \
      --segment-steps ${seg} \
      --warm-start-cycles ${warm} \
      --mode tf \
      --out "${OUT_DIR}/${out_name}.json"
    echo
}

# Warm-start matches K=1 training segment_steps=16 (continuous-cycle hidden).
WARM=16

# Apples-to-apples: fixed seg=2 across ALL horizons. h=12/24/40 share
# segment partitioning with h=1/4/6/8 in the multi-horizon slurm.
run_eval 12 2 ${WARM} "v3K1_step4000_h12_tf_warm${WARM}_seg2"
run_eval 24 2 ${WARM} "v3K1_step4000_h24_tf_warm${WARM}_seg2"
run_eval 40 2 ${WARM} "v3K1_step4000_h40_tf_warm${WARM}_seg2"

echo "ALL DONE"
ls -la "${OUT_DIR}/"
