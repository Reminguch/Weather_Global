#!/usr/bin/env bash
# v4.1 from-0: same trainer as the warm-start run but starts from random init.
# Same architecture (hidden=128 layers=1, ~308K params), same training config
# (K=4 closed-loop, lead-weighted loss, len_segment=32, bptt=8, lanes=4).
# Trains step 1 -> step 10000 (= 10K steps from scratch).
#
# Compare to warm-start (mz_fullmamba_v4_1_real.slurm: step 4001 -> 10000):
#   - Final step both 10000, so ckpts at step 10000 are directly comparable
#   - Warm-start has v3 K=1 markovian representation as starting point;
#     from-0 must learn that representation under K=4cl regime simultaneously
#   - If from-0 reaches similar/better RMSE Δ% in truth-anchored AR eval,
#     it's evidence that warm-start was trapping in K=1 markovian local optimum

#SBATCH --job-name=mz-fm-v41-from0
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=23:59:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v41_from0_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v41_from0_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v4_1/train_mz_fullmamba_v4_1.py
cd /home/lm8598/Weather_Global_experiments

RESULTS=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory

${PYTHON} -u ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${RESULTS}" \
  --run-name mz_fullmamba_v4_1_K1_K4cl_seg32_bptt8_lanes4_from0_to10k \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h \
  --len-segment 32 --bptt-steps 8 --num-lanes 4 \
  --target-steps-closed-loop 4 --loss-lead-weight-power 1.0 \
  --chunk-load-workers 6 --data-cache-mode auto --data-cache-max-gib 80 \
  --hidden-size 128 --layers 1 \
  --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --specialist-heads \
  --max-steps 10000 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 --lr 1e-4 \
  --baseline-precision fp32 --precision bf16
