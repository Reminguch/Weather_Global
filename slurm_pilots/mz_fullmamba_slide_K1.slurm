#!/usr/bin/env bash
# Sliding-window AR training of MZ-residual FullMamba head (K=1).
#
# This is the STANDARD AR training paradigm (matches GraphCast paper) — one
# random anchor per training sample, no S×K segment packing. Compared to our
# main K=1 ckpt (mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_4k_cont)
# which uses segment_steps=16 (16 anchors flattened into T=16 segment), this
# uses segment_steps=1 → each "segment" is one anchor → T = 1 × K = 1.
#
# Architecture is IDENTICAL to current K=1 best (FullMamba S6 d_state=16,
# expand=2, hidden=128, mesh_size=5, full 11 vars 1° + 13L). Only the data
# packing differs:
#   * Current K=1: T=16 per forward, SSM h propagates across 16 anchors
#   * Slide K=1:   T=1  per forward, SSM h fresh per anchor (no propagation)
#
# This isolates the contribution of multi-anchor packing in the original
# design. Expectation: slide K=1 likely underperforms current K=1 because
# T=1 gives Mamba SSM no temporal context (effectively reduces Mamba to a
# stateless MLP residual head).

#SBATCH --job-name=mz-fm-slide-K1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=08:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_slide_K1_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_slide_K1_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_slide/train_mz_fullmamba_slide.py
cd /home/lm8598/Weather_Global_experiments

DST_NAME=mz_fullmamba_slide_paperckpt_r1_in2_seg1_meshed_m5_h128_ds16_fullvars_K1_4k
DST_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/${DST_NAME}
mkdir -p "${DST_DIR}"

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \
  --run-name ${DST_NAME} \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h \
  --segment-steps 1 \
  --target-steps 1 --train-mode teacher \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --full-mamba --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --max-steps 4000 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16
