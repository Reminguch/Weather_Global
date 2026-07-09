#!/usr/bin/env bash
# res_mamba K=4 closed-loop at res=2 from GCv1 K=3.
# closed-loop = --residual-ar-feedback baseline_plus_residual:
#   AR-tail 4 steps each feed `baseline + residual` as next input (你描述的:
#     step 1 → corrected_1 → step 2 input = corrected_1 → ... → step 4)
# FULL-TRUNK LOSS (patch applied) — all 24 bptt steps contribute, not just AR tail.
# GC2 setup:
#   --init-from-baseline (strict=False): GC2 g2m + m2g + proc[0..1] overlaid from GC1.
#   --trainable-part proc_and_mamba: freeze g2m + m2g, train proc + Mamba.
#SBATCH --job-name=resmamba-GCv1K3-K4-closed
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --constraint=gpu80
#SBATCH --time=12:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/resmamba_GCv1K3_K4_closed_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/resmamba_GCv1K3_K4_closed_%j.err
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

$PYTHON -u -m src.models.mamba.training.segments_training \
  --model residual_mamba \
  --data-path "${DATA_PATH}" \
  --data-source prepared_array \
  --prepared-data-root "${PREPARED}" \
  --stats-dir "${STATS}" \
  --baseline-ckpt "${GCV1_K3}" \
  --out-dir "${ROOT}/results/resmamba_GCv1K3" \
  --run-name resmamba_GCv1K3_K4_closed_20k \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --resolution 2.0 --mesh-size 4 --width 512 --processor-msg-steps 2 \
  --input-duration 12h \
  --bptt-steps 24 --target-steps 4 --len-segment 48 \
  --batch-size 1 --eval-batch-size 1 \
  --eval-num-segments 4 --final-eval-num-segments 4 \
  --chunk-load-workers 6 \
  --batch-builder prepared_array \
  --data-cache-mode never \
  --max-steps 20000 --eval-every 1000 --checkpoint-every 1000 \
  --lr 1e-4 --weight-decay 1e-4 --precision bf16 \
  --temporal-backbone mamba \
  --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 1 \
  --temporal-stateful \
  --temporal-insert-count 2 \
  --residual-ar-feedback baseline_plus_residual \
  --init-from-baseline \
  --trainable-part proc_and_mamba \
  --memory-mode optimal \
  --seed 22

echo "resmamba GCv1K3 K=4 closed-loop 20k done"
