#!/usr/bin/env bash
# V-B1 smoke test (gpu-test QoS, higher priority).
# Goal: verify new feedback-clip code path runs end-to-end. Reach first eval at step 28500.
# Same config as full V-B1 c=0.5 but max-steps reduced so it fits gpu-test walltime.
# Writes to a separate run_name so it doesn't collide with the full run.
#SBATCH --job-name=resmamba-K2-vB1-smoke
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/resmamba_K2_vB1_smoke_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/resmamba_K2_vB1_smoke_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
RESUME_CKPT=${ROOT}/results/resmamba_GCv1K3/resmamba_GCv1K3_res2_K2_vA_smoke/ckpt_step28000.npz
RUN_NAME=resmamba_GCv1K3_res2_K2_vB1_smoke
OUT_DIR=${ROOT}/results/resmamba_GCv1K3_vB_smoke
mkdir -p ${OUT_DIR}

[ -f "${RESUME_CKPT}" ] || { echo "ERR: missing V-A step28000 ckpt"; exit 1; }

$PYTHON -u -m src.models.mamba.training.segments_training \
  --model residual_mamba \
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --data-source prepared_array \
  --prepared-data-root /scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/dataset/prepared_stream \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --baseline-ckpt "${GCV1_K3}" \
  --resume-ckpt "${RESUME_CKPT}" \
  --resume-step 28000 \
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
  --max-steps 28500 --eval-every 500 --checkpoint-every 1000 \
  --lr 1e-5 --weight-decay 1e-4 --precision bf16 \
  --temporal-backbone mamba \
  --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 1 \
  --temporal-stateful \
  --temporal-insert-count 2 \
  --residual-ar-feedback gated \
  --feedback-lambda-init 0.55 \
  --feedback-lambda-final 1.0 \
  --feedback-lambda-warmup-start-step 28000 \
  --feedback-lambda-warmup-end-step 32000 \
  --stop-grad-feedback \
  --feedback-clip-mode tanh \
  --feedback-clip-c 0.5 \
  --grad-clip-norm 0.5 \
  --residual-output-head enabled \
  --trainable-part proc_and_mamba \
  --memory-mode optimal \
  --seed 22

echo "V-B1 smoke done"
