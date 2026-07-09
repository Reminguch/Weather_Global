#!/usr/bin/env bash
# v4 K=1 stateful TBPTT smoke test.
#
# Recipe:
#   len_segment = 32, bptt_steps = 8, num_lanes = 4
#   -> each train step processes 4 lanes × 8 cycles = 32 K=1 predictions
#   -> hidden state carries across 4 chunks per segment, gradient cut at chunk
#      boundary (TBPTT-8 over 32-step rollout)
#   -> SSM weights see hidden depths {8, 16, 24, 32} during training
#
# Memory: T·B = 8·4 = 32 (matches v3 K=4 budget).
# Speed: NumpyBatchCache + 6 prefetch workers should give ~1-2s/step.

#SBATCH --job-name=mz-fm-v4-K1
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v4_K1_smoke_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v4_K1_smoke_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v4/train_mz_fullmamba_v4.py
cd /home/lm8598/Weather_Global_experiments

RESULTS=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory

${PYTHON} -u ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${RESULTS}" \
  --run-name mz_fullmamba_v4_K1_seg32_bptt8_lanes4_smoke \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h \
  --len-segment 32 --bptt-steps 8 --num-lanes 4 \
  --chunk-load-workers 6 --data-cache-mode auto --data-cache-max-gib 80 \
  --hidden-size 128 --layers 1 \
  --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --specialist-heads \
  --max-steps 200 --eval-every 100 --eval-max-segments 4 --checkpoint-every 100 \
  --grad-clip 1.0 --warmup-steps 50 --lr 1e-4 \
  --baseline-precision fp32 --precision bf16
