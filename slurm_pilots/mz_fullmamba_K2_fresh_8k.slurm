#!/usr/bin/env bash
# K=2 fresh training: resume from K=1 step 4000 (the converged K=1 ckpt) and
# train K=2 for 4000 K=2 steps continuously to step 8000. This is the CLEAN
# K=2 training: a single uninterrupted Adam state across all 4000 K=2 steps,
# no resume mid-K=2-phase.
#
# Replaces the previous K=2 plan (canceled job 7377586) which would have
# resumed from undertrained K=2 step 5000 -- that path had two Adam restarts
# (start of K=2 phase + mid-phase resume). This single-resume design matches
# K=1's training recipe (single Adam start, 4000 continuous steps) for fair
# K-curriculum comparison.
#
# After this finishes, K=4 / K=6 / K=8 will be re-trained fresh from K=2
# step 8000, each with its own 4000 K_K continuous steps.

#SBATCH --job-name=mz-fm-K2-fresh
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=20:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_K2_fresh_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_K2_fresh_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba/train_mz_fullmamba.py
cd /home/lm8598/Weather_Global_experiments

# K=1 converged ckpt (step 4000)
K1_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_4k_cont
DST_NAME=mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_8k_fresh
DST_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/${DST_NAME}
mkdir -p "${DST_DIR}"

# Pre-seed logs so 0..4000 (K=1 phase) plus 4000..8000 (clean K=2 phase) appear
# as one continuous curve. The new K=2 phase will entirely overwrite any
# previous K=2 attempt at the same step ranges.
cp "${K1_DIR}/train_log.json" "${DST_DIR}/train_log.json"
cp "${K1_DIR}/eval_log.json"  "${DST_DIR}/eval_log.json"

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \
  --run-name ${DST_NAME} \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h --segment-steps 16 \
  --target-steps 2 --train-mode target_rollout \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --full-mamba --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --max-steps 8000 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${K1_DIR}/mz_residual_step4000.pkl" \
  --resume-step 4000
