#!/usr/bin/env bash
# v24 SMOKE K=22: same big-Mamba config as smoke_K1 but with the full BPTT=24
# AR-tail. This is the actual memory stress test — K=1 single-step uses much
# less BPTT activation than K=22 closed-loop.
#
# Purpose: confirm that the K=22 train step (which holds 24 boundary saves of
# Mamba SSM state + conv cache) fits in 80GB GPU memory. NOT meant to produce
# a usable ckpt — runs from fresh init, will produce garbage.
#
# Expected runtime: ~25-40 minutes (JAX compile dominates; 2 K=22 steps).
# If this OOMs, the most actionable fix is dropping --temporal-hidden-size
# from 512 to 384 (saves 4× SSM state memory), then retrying.
#
# If smoke_K1 passes but smoke_K22 OOMs, this is the canonical memory
# constraint of the project (BPTT × Mamba state).
#SBATCH --job-name=v24-smoke-K22
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v24_smoke_K22_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v24_smoke_K22_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
# Low-risk JAX memory hygiene. NOT a real fix for OOM caused by Mamba BPTT
# state — these only reduce allocator/reservation overhead, not model memory.
# If OOM persists, the actionable next step is dropping --temporal-hidden-size.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_CPP_MIN_LOG_LEVEL=1
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v24/train_mz_v24.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RESIDUAL_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v24_smoke

${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name v24_smoke_K22_bigM \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 2 --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 24 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 22 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 512 \
  --temporal-d-state 16 --temporal-d-conv 8 --temporal-dt-rank auto \
  --temporal-layers 3
