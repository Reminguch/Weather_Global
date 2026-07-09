#!/usr/bin/env bash
# MZ at target_steps=2 on top of a K=2 FINE-TUNED baseline (matched training
# horizons -> apples-to-apples). Will wait for baseline FT (job 7248277) to
# finish via SLURM --dependency and then train for 400 gradient steps.

#SBATCH --job-name=mz-r4-K2m
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_r4_targetK2_matched_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_r4_targetK2_matched_%j.err
#SBATCH --dependency=afterok:7248277

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
  --baseline-ckpt /scratch/gpfs/DABANIN/lm8598/Weather_Global/lagrangian_lab/artifacts/checkpoints/long_iid_mamba/bfix_r4_m3_in32_t2_ft/ckpt_step28000.npz \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \
  --run-name mz_r4_m3_i32_seg32_h16_targetK2_matched \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 4 --mesh-size 3 \
  --input-duration 192h --segment-steps 32 \
  --target-steps 2 --train-mode target_rollout \
  --hidden-size 16 --layers 1 \
  --max-steps 400 --eval-every 100 --eval-max-segments 8 --checkpoint-every 200 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16
