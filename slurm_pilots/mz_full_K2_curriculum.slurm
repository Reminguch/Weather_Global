#!/usr/bin/env bash
# Curriculum Stage 1: K=1 -> K=2 rollout fine-tune.
# Resumes from 7288521 (meshed + 11var, K=1) step 2000, trains with
# target_steps=2 + target_rollout mode for 1000 additional steps.
# Dependency: waits for 7288521 to finish successfully.

#SBATCH --job-name=mz-K2-curriculum
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=06:00:00
#SBATCH --dependency=afterok:7288521
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_full_K2_curriculum_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_full_K2_curriculum_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
mkdir -p /home/lm8598/Weather_Global_experiments/results/mz_residual_memory
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mz/train_full_mz.py
cd /home/lm8598/Weather_Global_experiments

SRC_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_full_paperckpt_r1_in2_seg16_meshed_m5_h128_allvars_2k
DST_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_full_paperckpt_r1_in2_seg16_meshed_m5_h128_allvars_K2_cont
mkdir -p "${DST_DIR}"

# Pre-seed the new run-dir with the K=1 train/eval logs so the --resume-from
# path can append new entries to the full curve (K=1 history + K=2 ramp).
cp "${SRC_DIR}/train_log.json" "${DST_DIR}/train_log.json"
cp "${SRC_DIR}/eval_log.json" "${DST_DIR}/eval_log.json"

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \
  --run-name mz_full_paperckpt_r1_in2_seg16_meshed_m5_h128_allvars_K2_cont \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 \
  --input-duration 12h \
  --segment-steps 16 \
  --target-steps 2 --train-mode target_rollout \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --a-log-init -0.1 \
  --max-steps 3000 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${SRC_DIR}/mz_residual_step2000.pkl" \
  --resume-step 2000
