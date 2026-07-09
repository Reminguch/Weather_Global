#!/usr/bin/env bash
# Smoke test for gcmamba_v1 with the new PreparedArrayStore data loader
# (replaces the OOM-prone NumpyBatchCache+zarr from run01).
# Same model/training config as run01, but max-steps=100 to verify the
# whole pipeline runs end-to-end in <30 min before kicking off the 10k-step
# production rerun.
#SBATCH --job-name=gcm-v1-run02-smoke
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/gcm_v1_run02_smoke_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/gcm_v1_run02_smoke_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/gcmamba_v1
mkdir -p ${OUT_DIR}

$PYTHON -u scripts/training/gcmamba_v1/train_gcmamba_v1.py \
  --out-dir ${OUT_DIR} \
  --run-name gcmamba_v1_run02_prepared_smoke \
  --resolution 1.0 --mesh-size 5 --width 512 --processor-msg-steps 16 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 --temporal-d-state 16 \
  --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 1 \
  --temporal-stateful \
  --bptt-steps 24 --target-steps 1 --len-segment 96 \
  --batch-size 1 \
  --lr 1e-4 --weight-decay 1e-4 --grad-clip 1.0 \
  --warmup-steps 50 \
  --max-steps 100 --eval-every 50 --checkpoint-every 100 \
  --train-start-year 2015 --train-end-year 2021 \
  --val-year 2022 \
  --seed 22 \
  --memory-mode optimal

echo "smoke run02 done"
