#!/usr/bin/env bash
# gcmamba_v1 rolling_ar smoke: bptt=8, target-steps=4 (= v22/Ilya closed-loop).
# Validates the AR feedback path and the loss-on-AR-tail aggregation.
#SBATCH --job-name=gcmamba-v1-rollar
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --constraint=gpu80
#SBATCH --time=02:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/gcmamba_v1_rollar_smoke_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/gcmamba_v1_rollar_smoke_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/gcmamba_v1
mkdir -p ${OUT_DIR}

# bptt=8, target-steps=4 -> 4 truth-fed steps + 4 AR steps (loss on AR tail).
# Full BPTT through AR chain (matches v22 default).
$PYTHON -u scripts/training/gcmamba_v1/train_gcmamba_v1.py \
  --out-dir ${OUT_DIR} \
  --run-name rollar_bptt8_tgt4_di64_h128_l1 \
  --resolution 1.0 --mesh-size 5 --width 512 --processor-msg-steps 16 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 64 --temporal-d-state 16 \
  --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 1 \
  --temporal-stateful \
  --bptt-steps 8 --target-steps 4 --len-segment 32 \
  --batch-size 1 \
  --lr 1e-4 --weight-decay 1e-4 --grad-clip 1.0 \
  --max-steps 100 --eval-every 50 --checkpoint-every 50 \
  --seed 0
echo "gcmamba_v1 rolling_ar smoke done"
