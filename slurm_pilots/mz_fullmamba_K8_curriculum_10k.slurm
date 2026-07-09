#!/usr/bin/env bash
# K=8 curriculum (48h horizon, day 2): resume from FullMamba K=6 step 8000
# (just plateaued at TF_w +0.53%, AR_w +0.35%) and continue with K=8
# target_rollout for ~2000 more steps to step 10000. This is the next natural
# curriculum increment K=4 -> K=6 -> K=8 toward paper's K=12 endpoint.
#
# Memory budget at segment_steps=4, target_steps=8:
#   T = 4 x 8 = 32 positions per segment (same as our successful K=4 seg=8 setup)
#   ~10 GB activation + ~5 GB baseline = ~15 GB peak (very safe on 80GB A100)
#
# segment_steps=4 (vs 8) gives fewer anchors per segment but keeps T=32 within
# the known-good envelope. K=8 is in-distribution at horizon=8 (48h, day 2).
# No architecture change vs K=6 ckpt -> strict resume.

#SBATCH --job-name=mz-fm-K8-10k
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=20:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_K8_10k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_K8_10k_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba/train_mz_fullmamba.py
cd /home/lm8598/Weather_Global_experiments

SRC_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K6_8k
DST_NAME=mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K8_10k
DST_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/${DST_NAME}
mkdir -p "${DST_DIR}"

# Pre-seed logs so K=1->K=2->K=4->K=6 (steps 0..8000) followed by K=8 (8000..10000)
# appear as one continuous curve.
cp "${SRC_DIR}/train_log.json" "${DST_DIR}/train_log.json"
cp "${SRC_DIR}/eval_log.json"  "${DST_DIR}/eval_log.json"

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \
  --run-name ${DST_NAME} \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h --segment-steps 4 \
  --target-steps 8 --train-mode target_rollout \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --full-mamba --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --max-steps 10000 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${SRC_DIR}/mz_residual_step8000.pkl" \
  --resume-step 8000
