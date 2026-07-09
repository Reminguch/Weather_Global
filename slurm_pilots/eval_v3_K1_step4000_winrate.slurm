#!/usr/bin/env bash
# v3 K=1 step 4000 TF h=1 eval — fills the K=1 row in the K-curriculum
# per-target win-rate table:
#
#   ckpt                       K   targets    RMSE win-rate
#   ---------------------------------------------------------
#   v3 K=1 step 4000  (this)   1   1*83=83    TBD
#   v3 K=2 step 5500           2   2*83=166   166/166 = 100% (existing)
#   v3 K=4 step 6000           4   4*83=332   331/332 = 99.7% (existing)
#   v3 K=4 step 7000           4   4*83=332   328/332 = 98.8% (existing)
#   v3 K=6 step 8500           6   6*83=498   TBD (queued in 7558339 NORMAL)

#SBATCH --job-name=eval-K1-wr
#SBATCH --account=dabanin
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_K1_wr_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_K1_wr_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/eval_per_level.py
cd /home/lm8598/Weather_Global_experiments

V3K1_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v3_PARITY_K1_3500to6k
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-05-01_v3_winrate
mkdir -p "${OUT_DIR}"

${PYTHON} ${SCRIPT} \
  --run-dir "${V3K1_DIR}" \
  --step 4000 \
  --num-segments 64 \
  --horizon 1 \
  --mode tf \
  --out "${OUT_DIR}/v3K1_step4000_h1_tf.json"

echo "DONE."
ls -la "${OUT_DIR}/"
