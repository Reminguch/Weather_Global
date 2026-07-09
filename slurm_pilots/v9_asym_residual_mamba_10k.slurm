#!/usr/bin/env bash
# v9 real run: 10K steps with ckpt every 500 (20 ckpts × ~443MB = 8.9GB on scratch).
# Smoke 7724191 verified architecture: init loss 0.84, grad_norm 0.07, no NaN/OOM.
# Same hyperparams as smoke; only max_steps + checkpoint_every change.

#SBATCH --job-name=v9-arm-10k
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=16:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v9_arm_10k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v9_arm_10k_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v9/train_mz_v9.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v9_asym_residual_mamba

${PYTHON} -u ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name v9_arm_10k \
  --batch-size 1 \
  --max-steps 10000 --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 0 --precision bf16 \
  --input-duration 12h \
  --target-steps 1 \
  --sequential-segment-steps 32 --bptt-steps 8 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2 \
  --trainable-part mamba
