#!/usr/bin/env bash
#SBATCH --job-name=mz-r6-infer
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_r6_infer_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_r6_infer_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/infer_mz_save_tensors.py

cd /home/lm8598/Weather_Global_experiments

${PYTHON} ${SCRIPT} \
  --run-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_r6_m3_i32_seg32_h16_teacher_step10000 \
  --step 200 \
  --num-segments 4 \
  --out /home/lm8598/Weather_Global_experiments/results/2026-04-22_mz_r4_r6_comparison/2026-04-22_mz_r6_step10000_analysis/tensors_step200_firstsegs.npz
