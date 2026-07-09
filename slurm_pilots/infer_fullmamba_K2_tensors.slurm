#!/usr/bin/env bash
# Run the FullMamba + K=2 model (7318997 step 3000) forward on 8 eval segments
# and dump baseline/corrected/truth tensors for plotting.

#SBATCH --job-name=mz-infer-K2
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/infer_K2_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/infer_K2_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/infer_mz_meshed_save_tensors.py
cd /home/lm8598/Weather_Global_experiments

RUN_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_5k
OUT=/home/lm8598/Weather_Global_experiments/results/2026-04-24_fullmamba_K2_tensors/tensors_step3000.npz

mkdir -p "$(dirname ${OUT})"

${PYTHON} ${SCRIPT} \
  --run-dir ${RUN_DIR} \
  --step 3000 \
  --num-segments 8 \
  --out ${OUT} \
  --locations "Beijing:39.9:116.4,NYC:40.7:286.0,Sydney:-33.9:151.2,Nairobi:-1.3:36.8,London:51.5:359.9,Tokyo:35.7:139.7,LA:34.0:241.7,Mumbai:19.1:72.9"
