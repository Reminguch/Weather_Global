#!/usr/bin/env bash
#SBATCH --job-name=mz-infer-weak
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_infer_weak_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_infer_weak_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/infer_mz_save_tensors.py

cd /home/lm8598/Weather_Global_experiments

${PYTHON} ${SCRIPT} \
  --run-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_r4_m3_i32_seg32_h16_fullnorm_step5000 \
  --step 400 \
  --num-segments 4 \
  --out /home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_r4_m3_i32_seg32_h16_fullnorm_step5000/tensors_step400_firstsegs.npz
