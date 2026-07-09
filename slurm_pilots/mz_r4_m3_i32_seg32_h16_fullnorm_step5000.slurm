#!/usr/bin/env bash
# Apples-to-apples companion run: MZ at res=4 but using baseline ckpt_step5000
# (the same checkpoint-step used for res=2/6/8), instead of the original
# mz_r4 run that used ckpt_step23000.
#SBATCH --job-name=mz-r4-fn-s5k
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_r4_m3_i32_seg32_h16_fullnorm_step5000_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_r4_m3_i32_seg32_h16_fullnorm_step5000_%j.err

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
  --baseline-ckpt /scratch/gpfs/DABANIN/lm8598/Weather_Global/lagrangian_lab/artifacts/checkpoints/long_iid_mamba/bfix_r4_m3_in32_t1/ckpt_step5000.npz \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \
  --run-name mz_r4_m3_i32_seg32_h16_fullnorm_step5000 \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 4 --mesh-size 3 \
  --input-duration 192h --segment-steps 32 \
  --hidden-size 16 --layers 1 \
  --max-steps 400 --eval-every 100 --eval-max-segments 8 --checkpoint-every 200 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16
