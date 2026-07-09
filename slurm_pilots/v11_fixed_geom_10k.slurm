#!/usr/bin/env bash
# v11 full run: v3-style fixed geometric grid<->mesh + Mamba, 10K steps.
# Smoke (7751782) confirmed: step 500 = 0.747, matches v9/v10 within noise
# (~0.74) with only 151K trainable params (44x smaller than v10's 6.7M).
# Now run full 10K from scratch (same seed) to see whether 0.74 is the floor
# or whether longer training breaks through.

#SBATCH --job-name=v11-fix-10k
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=12:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v11_fix_10k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v11_fix_10k_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v11/train_mz_v11.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v11_fixed_geom_mamba

${PYTHON} -u ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --resolution 1.0 --mesh-size 5 --hidden-size 128 \
  --baseline-msg-steps 16 \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name v11_fix_10k \
  --batch-size 1 \
  --max-steps 10000 --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 0 --precision bf16 \
  --input-duration 12h \
  --target-steps 1 \
  --sequential-segment-steps 32 --bptt-steps 8 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2
