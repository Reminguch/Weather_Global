#!/usr/bin/env bash
# Smoke test for train_mz_residual_memory_resume.py:
# Resume paper h=128 run from its step 400 checkpoint, train 10 more steps (to
# step 410). Should produce a non-trivial loss delta and valid checkpoints.
# Writes to a NEW run-name so the original run_dir is untouched.

#SBATCH --job-name=mz-resume-smk
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_resume_smoke_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_resume_smoke_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
mkdir -p /home/lm8598/Weather_Global_experiments/results/mz_residual_memory
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/train_mz_residual_memory_resume.py
cd /home/lm8598/Weather_Global_experiments

# Pre-seed the new run-dir with the train_log / eval_log from step 400 so the
# resume script has something to append to.
SRC_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_paperckpt_r1_in2_seg16_h128_teacher
DST_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_resume_smoke_h128
mkdir -p "${DST_DIR}"
cp "${SRC_DIR}/train_log.json" "${DST_DIR}/"
cp "${SRC_DIR}/eval_log.json" "${DST_DIR}/"

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \
  --run-name mz_resume_smoke_h128 \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 \
  --input-duration 12h \
  --segment-steps 16 \
  --target-steps 1 --train-mode teacher \
  --hidden-size 128 --layers 1 \
  --max-steps 410 --eval-every 5 --eval-max-segments 2 --checkpoint-every 5 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${SRC_DIR}/mz_residual_step400.pkl" \
  --resume-step 400
