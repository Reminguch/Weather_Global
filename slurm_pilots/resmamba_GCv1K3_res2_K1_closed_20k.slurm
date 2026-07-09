#!/usr/bin/env bash
# res_mamba (architecture B): frozen GCv1-K3 baseline + residual head GC2.
# v22-style: --temporal-location mesh_processor_interleaved, insert_count=2.
# Closed-loop training: --residual-ar-feedback baseline_plus_residual (v22closed-style).
#
# *** GC2 freeze setup (user request 2026-06-15) ***
#   --init-from-baseline: overlay GC1 (GCv1 K=3) → GC2 at fresh init
#                         so GC2's g2m/m2g/proc start identical to GC1.
#   --trainable-part proc_and_mamba:
#       freeze GC2's g2m + m2g (= identical to GC1 forever)
#       train  GC2's processor (mp=2) + Mamba
#   → trainable = 3.94M (proc) + 0.43M (Mamba)  = 4.37M
#   → frozen    = GC1 (17.59M) + GC2.g2m + GC2.m2g (6.29M) = 23.88M
#
# K=1 truth-fed (no AR cascade), bptt=24, layers=1, d_state=16 (same Mamba as gc_mamba run).
#SBATCH --job-name=resmamba-GCv1K3-K1-closed
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --constraint=gpu80
#SBATCH --time=12:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/resmamba_GCv1K3_K1_closed_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/resmamba_GCv1K3_K1_closed_%j.err
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
  --model residual_mamba \
  --data-path "${DATA_PATH}" \
  --data-source prepared_array \
  --prepared-data-root "${PREPARED}" \
  --stats-dir "${STATS}" \
  --baseline-ckpt "${GCV1_K3}" \
  --out-dir "${ROOT}/results/resmamba_GCv1K3" \
  --run-name resmamba_GCv1K3_res2_K1_closed_20k \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --resolution 2.0 --mesh-size 4 --width 512 --processor-msg-steps 2 \
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

echo "resmamba GCv1K3 res=2 K=1 closed-loop 20k done"
