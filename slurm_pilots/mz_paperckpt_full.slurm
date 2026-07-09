#!/usr/bin/env bash
# Full MZ training on public DeepMind GraphCast_small ckpt (1° grid, mesh=5,
# width=512, 16 msg steps, input=12h). 400 training steps, teacher-forced,
# h=16 MZ residual on top of a frozen paper baseline.

#SBATCH --job-name=mz-paper-full
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_paperckpt_full_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_paperckpt_full_%j.err

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
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \
  --run-name mz_paperckpt_r1_in2_seg16_h16_teacher \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 \
  --input-duration 12h \
  --segment-steps 16 \
  --target-steps 1 --train-mode teacher \
  --hidden-size 16 --layers 1 \
  --max-steps 400 --eval-every 100 --eval-max-segments 8 --checkpoint-every 200 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16
