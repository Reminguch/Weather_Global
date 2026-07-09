#!/usr/bin/env bash
# MZ residual on r=2 input=2 baseline (v2_metrics/baseline_20k, step20000),
# K=1 teacher, hidden=16, segment_steps sweep. This file: seg=16.

#SBATCH --job-name=mz-r2in2-s48
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=06:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_r2in2_s48_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_r2in2_s48_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
mkdir -p /home/lm8598/Weather_Global_experiments/results/mz_residual_memory

export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/train_mz_residual_memory.py

cd /home/lm8598/Weather_Global_experiments

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt /scratch/gpfs/DABANIN/lm8598/Weather_Global/artifacts/checkpoints/v2_metrics/baseline_20k/ckpt_step20000.npz \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \
  --run-name mz_r2_in2_seg48_h16_teacher_b20k \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 2 --mesh-size 3 \
  --input-duration 12h --segment-steps 48 \
  --target-steps 1 --train-mode teacher \
  --hidden-size 16 --layers 1 \
  --max-steps 400 --eval-every 100 --eval-max-segments 8 --checkpoint-every 200 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16
