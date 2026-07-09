#!/usr/bin/env bash
# Sliding-window K=2 AR mode training. Resume from slide K=1 step 4000 ckpt
# and train 4000 K=2 steps (to step 8000) with FULLY AUTOREGRESSIVE training
# (train_mode="ar", teacher_forcing_prob=0).
#
# This continues the standard-AR-paradigm chain:
#   K=1 (slide, segment=1, T=1, 4000 steps)  → 7391546
#   K=2 (slide, segment=1, T=2, 4000 K=2 steps, AR train) ← THIS
#
# segment_steps=1 → each training sample is 1 anchor + K=2 step rollout.
# train_mode=ar  → no truth_residual at any position; model is forced to
#                   train on its own self-fed r̂ (paper-style "open-ended AR").
#
# Architecture identical to slide K=1: FullMamba S6 (d_state=16, expand=2),
# hidden=128, mesh=5, full 11 vars 1° + 13L.
#
# Dependency: waits for 7391546 (slide K=1) to finish OK.

#SBATCH --job-name=mz-fm-slide-K2-ar
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=12:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_slide_K2_ar_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_slide_K2_ar_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_slide/train_mz_fullmamba_slide.py
cd /home/lm8598/Weather_Global_experiments

SRC_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_slide_paperckpt_r1_in2_seg1_meshed_m5_h128_ds16_fullvars_K1_4k
DST_NAME=mz_fullmamba_slide_paperckpt_r1_in2_seg1_meshed_m5_h128_ds16_fullvars_K2_8k_ar
DST_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/${DST_NAME}
mkdir -p "${DST_DIR}"

# Pre-seed logs from K=1 slide ckpt so 0..4000 (K=1 slide) + 4000..8000 (K=2 ar)
# appear as continuous trajectory.
cp "${SRC_DIR}/train_log.json" "${DST_DIR}/train_log.json"
cp "${SRC_DIR}/eval_log.json"  "${DST_DIR}/eval_log.json"

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \
  --run-name ${DST_NAME} \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h \
  --segment-steps 1 \
  --target-steps 2 --train-mode ar \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --full-mamba --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --max-steps 8000 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 100 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${SRC_DIR}/mz_residual_step4000.pkl" \
  --resume-step 4000
