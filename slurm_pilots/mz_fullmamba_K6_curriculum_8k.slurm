#!/usr/bin/env bash
# K=6 curriculum (36h horizon, day 1.5): resume from FullMamba K=4 step 6000
# (which just plateaued — see eval_log step 5400-6000) and continue with K=6
# target_rollout for ~2000 more steps to step 8000. This is the next natural
# curriculum increment after K=4; halfway between paper's K=12 endpoint and
# the K=4 we just finished.
#
# Memory budget at segment_steps=8, target_steps=6:
#   T = 8 × 6 = 48 positions per segment
#   ~14 GB activation + ~5 GB baseline = ~20 GB peak (well under 80 GB A100)
#
# No architecture change vs K=4 ckpt -> strict resume (no --allow-partial-resume).
# Independent of the K=2+processor line (job 7363554), runs in parallel after
# the horizon=4 eval (job 7364671) frees its GPU.

#SBATCH --job-name=mz-fm-K6-8k
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=20:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_K6_8k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_K6_8k_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba/train_mz_fullmamba.py
cd /home/lm8598/Weather_Global_experiments

SRC_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K4_7k
DST_NAME=mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K6_8k
DST_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/${DST_NAME}
mkdir -p "${DST_DIR}"

# Pre-seed logs so K=1->K=2->K=4 (steps 0..6000) followed by K=6 (steps 6000..8000)
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
  --resolution 1.0 --mesh-size 5 --input-duration 12h --segment-steps 8 \
  --target-steps 6 --train-mode target_rollout \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --full-mamba --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --max-steps 8000 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${SRC_DIR}/mz_residual_step6000.pkl" \
  --resume-step 6000
