#!/usr/bin/env bash
# K=2 open-loop residual_mamba pretrain on GCv1 K=3 baseline (res=2).
# Fresh init: GCv1 K=3 weights overlaid into g2m/m2g/proc; Mamba zero-init.
# Training: target_steps=2 (1 truth + 1 AR tail step). Feedback: baseline only.
#SBATCH --job-name=resmamba-K2-v22inp-smoke
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/resmamba_K2_v22inp_smoke_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/resmamba_K2_v22inp_smoke_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
RUN_NAME=resmamba_GCv1K3_res2_K2_v22inp_smoke
OUT_DIR=${ROOT}/results/resmamba_GCv1K3_openloop_Kscan_smoke
mkdir -p ${OUT_DIR}

$PYTHON -u -m src.models.mamba.training.segments_training \
  --model residual_mamba \
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --data-source prepared_array \
  --prepared-data-root /scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/dataset/prepared_stream \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --baseline-ckpt "${GCV1_K3}" \
  --out-dir "${OUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --resolution 2.0 --mesh-size 4 --width 512 --processor-msg-steps 2 \
  --input-duration 12h \
  --bptt-steps 24 --target-steps 2 --len-segment 48 \
  --batch-size 1 --eval-batch-size 1 \
  --eval-num-segments 4 --final-eval-num-segments 4 \
  --chunk-load-workers 6 \
  --batch-builder prepared_array \
  --data-cache-mode never \
  --max-steps 1500 --eval-every 1000 --checkpoint-every 2000 \
  --lr 1e-4 --weight-decay 1e-4 --precision bf16 \
  --temporal-backbone mamba \
  --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --temporal-stateful \
  --residual-ar-feedback baseline \
  --grad-clip-norm 1.0 \
  --residual-output-head enabled \
  --trainable-part all \
  --init-from-baseline \
  --memory-mode optimal \
  --seed 22

echo "K=2 open-loop pretrain done"
