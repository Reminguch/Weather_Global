#!/usr/bin/env bash
# v3 K=1 step 4000 ckpt @ h=1 and h=4, TF, seg=2, warm=16.
# Companion to eval_v3_K1_step4000_h6h8.slurm (which is already running);
# together fills the K=1 row at h={1, 4, 6, 8} in the multi-horizon table.
#
# Bug fix in eval_per_level.py: when --warm-start-cycles > 0, force
# effective_anchor_as_batch=True regardless of horizon, so warm + forecast
# phases share the [T, B=S] layout. Previously at h=1 the legacy fallback
# (B=1) collided with warm phase's anchor-as-batch (B=S) -> shape mismatch.

#SBATCH --job-name=eval-K1-h1h4
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_K1_h1h4_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_K1_h1h4_%j.err

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

WARM=16

run_eval 1 2 ${WARM} "v3K1_step4000_h1_tf_warm${WARM}_seg2"
run_eval 4 2 ${WARM} "v3K1_step4000_h4_tf_warm${WARM}_seg2"

echo "ALL DONE"
ls -la "${OUT_DIR}/"
