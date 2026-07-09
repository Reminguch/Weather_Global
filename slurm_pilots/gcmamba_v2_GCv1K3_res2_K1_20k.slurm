#!/usr/bin/env bash
# GC_mamba (architecture A): Mamba inside frozen GCv1-K3 res=2 processor.
# Baseline = GCv1 K=3 ckpt (frozen via --trainable-part=mamba).
# Mamba inserted post-processor (mesh_post_processor, insert=1).
# K=1 truth-fed (no AR), 20000 steps, bptt=24, layers=1, d_state=16.
#SBATCH --job-name=gcmv2-GCv1K3-res2-K1
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --constraint=gpu80
#SBATCH --time=12:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/gcmv2_GCv1K3_res2_K1_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/gcmv2_GCv1K3_res2_K1_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
PREPARED=/scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/dataset/prepared_stream
STATS=${ROOT}/data/graphcast/graphcast/stats
DATA_PATH=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz

if [ ! -f "${GCV1_K3}" ]; then
    echo "ERROR: GCv1 K=3 ckpt not found: ${GCV1_K3}" >&2; exit 1
fi

$PYTHON -u -m src.models.mamba.training.segments_training \
  --model gc_mamba \
  --data-path "${DATA_PATH}" \
  --data-source prepared_array \
  --prepared-data-root "${PREPARED}" \
  --stats-dir "${STATS}" \
  --ckpt-in "${GCV1_K3}" \
  --init-from-graphcast-ckpt "${GCV1_K3}" \
  --out-dir "${ROOT}/results/gcmamba_v2" \
  --run-name gcmamba_v2_GCv1K3_res2_K1_20k \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --resolution 2.0 --mesh-size 4 --width 512 --processor-msg-steps 6 \
  --input-duration 12h \
  --bptt-steps 24 --target-steps 1 --len-segment 48 \
  --batch-size 1 --eval-batch-size 1 \
  --eval-num-segments 4 --final-eval-num-segments 4 \
  --chunk-load-workers 6 \
  --batch-builder prepared_array \
  --data-cache-mode never \
  --max-steps 20000 --eval-every 1000 --checkpoint-every 1000 \
  --lr 1e-4 --weight-decay 1e-4 --precision bf16 \
  --temporal-backbone mamba \
  --temporal-location mesh_post_processor \
  --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 1 \
  --temporal-stateful \
  --temporal-insert-count 1 \
  --zero-init-temporal-out \
  --trainable-part mamba \
  --memory-mode optimal \
  --seed 22

echo "gcmamba_v2 GCv1K3 res=2 K=1 20k done"
