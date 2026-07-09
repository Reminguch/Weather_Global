#!/usr/bin/env bash
# v9 corrected: train ALL of GC2 + Mamba + zero-init head end-to-end.
# Previous run (7729563) only trained 46 Mamba/head leaves with 94 frozen
# GC layers — Mamba's modifications to mesh latents got washed by the next
# frozen mesh_gnn step. Loss plateaued at ~0.78 from step 2000 onward.
#
# This run: --trainable-part all → all 11.09M residual_params get gradients.
# Same lr=1e-4 single optimizer; baseline GC1 (DeepMind) stays untouched
# (it's in baseline_params, not residual_params).

#SBATCH --job-name=v9-full-10k
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=16:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v9_full_10k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v9_full_10k_%j.err

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
  --run-name v9_full_train_10k \
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
  --trainable-part all
