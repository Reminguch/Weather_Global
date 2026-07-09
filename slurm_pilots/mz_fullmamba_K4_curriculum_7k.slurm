#!/usr/bin/env bash
# FullMamba + K=4 curriculum: resume 7318997 (FullMamba K=2, step 5000) and
# train K=4 rollout for 2000 more steps (→ step 7000). Continues the
# progressive K=1 → K=2 → K=4 curriculum on the same FullMamba weights.
# Dependency: waits for 7318997 to finish successfully.

#SBATCH --job-name=mz-fm-K4-7k
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=20:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_K4_7k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_K4_7k_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba/train_mz_fullmamba.py
cd /home/lm8598/Weather_Global_experiments

SRC_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_5k
DST_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K4_7k
mkdir -p "${DST_DIR}"

# Pre-seed logs so step 0-5000 (K=1: 0-2000, K=2: 2000-5000) + new K=4 steps
# 5200..7000 appear as one continuous curve.
cp "${SRC_DIR}/train_log.json" "${DST_DIR}/train_log.json"
cp "${SRC_DIR}/eval_log.json" "${DST_DIR}/eval_log.json"

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \
  --run-name mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K4_7k \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h --segment-steps 8 \
  --target-steps 4 --train-mode target_rollout \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --full-mamba --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --max-steps 7000 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${SRC_DIR}/mz_residual_step5000.pkl" \
  --resume-step 5000
