#!/usr/bin/env bash
# FullMamba + full-variables 2K: paper GraphCast_small baseline + meshed m=5 +
# h=128 + d_state=16 input-dep B/C + all 11 GraphCast target variables (F=83).
# Direct A/B against 7288521 (same config, simplified SelectiveSSMBlock).
# Only the Mamba block differs.

#SBATCH --job-name=mz-fm-fv-2k
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=09:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fullmamba_fullvars_2k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fullmamba_fullvars_2k_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
mkdir -p /home/lm8598/Weather_Global_experiments/results/mz_residual_memory
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba/train_mz_fullmamba.py
cd /home/lm8598/Weather_Global_experiments

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \
  --run-name mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_2k \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 \
  --input-duration 12h \
  --segment-steps 16 \
  --target-steps 1 --train-mode teacher \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --full-mamba --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --max-steps 2000 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16
