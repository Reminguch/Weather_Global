#!/usr/bin/env bash
# GCmamba_v2 smoke at K=1 (truth-fed everywhere, rolling_ar=False).
# K=1 goes through the non-rolling loss branch (already full-trunk mean
# by construction; my patch matters only for K>1, but we want a K=1
# baseline first per Ilya's curriculum: K=1 → K=4 → K=8).
#
# Config: bptt=24, layers=1, d_state=16, len_segment=48, memory_mode=optimal,
# data-source=prepared_array (memmap, no RAM cache).
#SBATCH --job-name=gcmamba-v2-K1-smoke
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/gcmamba_v2_K1_smoke_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/gcmamba_v2_K1_smoke_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DATA_PATH=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
PREPARED=${ROOT}/data/graphcast/graphcast/dataset/prepared_stream
STATS=${ROOT}/data/graphcast/graphcast/stats
CKPT_IN="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"

OUT_DIR=${ROOT}/results/gcmamba_v2
mkdir -p ${OUT_DIR}

$PYTHON -u -m src.models.mamba.training.segments_training \
  --model gc_mamba \
  --data-path "${DATA_PATH}" \
  --data-source prepared_array \
  --prepared-data-root "${PREPARED}" \
  --stats-dir "${STATS}" \
  --ckpt-in "${CKPT_IN}" \
  --out-dir "${OUT_DIR}" \
  --run-name gcmamba_v2_bptt24_K1_L1_ds16_smoke \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --width 512 --processor-msg-steps 16 \
  --input-duration 12h \
  --bptt-steps 24 --target-steps 1 --len-segment 48 \
  --batch-size 1 --eval-batch-size 1 \
  --eval-num-segments 4 --final-eval-num-segments 4 \
  --chunk-load-workers 6 \
  --batch-builder prepared_array \
  --data-cache-mode never \
  --max-steps 250 --eval-every 50 --checkpoint-every 100 \
  --lr 1e-4 --weight-decay 1e-4 --precision bf16 \
  --temporal-backbone mamba \
  --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 1 \
  --temporal-stateful \
  --temporal-insert-count 2 \
  --zero-init-temporal-out \
  --trainable-part mamba \
  --memory-mode optimal \
  --seed 22

echo "gcmamba_v2 K=1 smoke done"
