#!/usr/bin/env bash
# V-B fix Smoke A — RECOVERY test WITH stream consistency.
# Resume from B route bad basin step31000 (rotating=5.97, λ=1 unclipped trained).
# Apply clip c=0.5 + consistency fix (Mamba memory uses λ*clipped_residual,
# matching the physical input). Hypothesis: recovery should be at least as good
# as buggy V-B1 smoke A (which dropped 5.97→2.32 in 500 steps).
#SBATCH --job-name=resmamba-K2-vBfix-smokeA-recovery
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/resmamba_K2_vBfix_smokeA_recovery_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/resmamba_K2_vBfix_smokeA_recovery_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
RESUME_CKPT=${ROOT}/results/resmamba_GCv1K3_ablation/resmamba_GCv1K3_res2_K2_vA_stageC_fast_lam1_plateau/ckpt_step31000.npz
RUN_NAME=resmamba_GCv1K3_res2_K2_vBfix_smokeA_recovery
OUT_DIR=${ROOT}/results/resmamba_GCv1K3_vBfix
mkdir -p ${OUT_DIR}

[ -f "${RESUME_CKPT}" ] || { echo "ERR: missing B route bad-basin ckpt"; exit 1; }

$PYTHON -u -m src.models.mamba.training.segments_training \
  --model residual_mamba \
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --data-source prepared_array \
  --prepared-data-root /scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/dataset/prepared_stream \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --baseline-ckpt "${GCV1_K3}" \
  --resume-ckpt "${RESUME_CKPT}" \
  --resume-step 31000 \
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
  --max-steps 35000 --eval-every 500 --checkpoint-every 1000 \
  --lr 1e-5 --weight-decay 1e-4 --precision bf16 \
  --temporal-backbone mamba \
  --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 1 \
  --temporal-stateful \
  --temporal-insert-count 2 \
  --residual-ar-feedback gated \
  --feedback-lambda-init 1.0 \
  --feedback-lambda-final 1.0 \
  --feedback-lambda-warmup-start-step 31000 \
  --feedback-lambda-warmup-end-step 31000 \
  --stop-grad-feedback \
  --feedback-clip-mode tanh \
  --feedback-clip-c 0.5 \
  --grad-clip-norm 0.5 \
  --residual-output-head enabled \
  --trainable-part proc_and_mamba \
  --memory-mode optimal \
  --seed 22

echo "V-B fix smokeA done: bad basin recovery with consistent streams"
