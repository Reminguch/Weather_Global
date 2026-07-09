#!/usr/bin/env bash
# v3 K=1 step 4000 ckpt @ h=6 and h=8, TF, seg=2, warm=16.
# Fills the K=1 row at h=6 and h=8 in the K=1-vs-K=6 multi-horizon table.
# Same flags (seg=2, warm=16) as the K=1 long-horizon slurm that succeeded
# at h=12/24/40, so this should also succeed (h>=2 -> anchor-as-batch
# layout, avoiding the h=1 legacy-layout shape bug).

#SBATCH --job-name=eval-K1-h6h8
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_K1_h6h8_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_K1_h6h8_%j.err

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

run_eval 6 2 ${WARM} "v3K1_step4000_h6_tf_warm${WARM}_seg2"
run_eval 8 2 ${WARM} "v3K1_step4000_h8_tf_warm${WARM}_seg2"

echo "ALL DONE"
ls -la "${OUT_DIR}/"
