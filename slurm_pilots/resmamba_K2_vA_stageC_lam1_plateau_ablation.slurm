#!/usr/bin/env bash
# V-A Stage C ablation B: λ=1.0 fixed plateau from old fast Stage C step30000 ckpt.
# Question: does λ=1 settle if given enough plateau time (10k steps), or did fast
#           ramp push the model into a bad closed-loop basin that can't recover?
# Independent of slow-ramp A path (different out_dir, different run_name).
#SBATCH --job-name=resmamba-K2-vA-lam1-plateau
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=08:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/resmamba_K2_vA_lam1_plateau_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/resmamba_K2_vA_lam1_plateau_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
# Resume from old fast Stage C step 30000 (λ=1.0 reached point, rotating=4.88)
RESUME_CKPT=${ROOT}/results/resmamba_GCv1K3/resmamba_GCv1K3_res2_K2_vA_smoke/ckpt_step30000.npz
RUN_NAME=resmamba_GCv1K3_res2_K2_vA_stageC_fast_lam1_plateau
OUT_DIR=${ROOT}/results/resmamba_GCv1K3_ablation
mkdir -p ${OUT_DIR}

[ -f "${RESUME_CKPT}" ] || { echo "ERR: missing fast Stage C step30000 ckpt"; exit 1; }

$PYTHON -u -m src.models.mamba.training.segments_training \
  --model residual_mamba \
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --data-source prepared_array \
  --prepared-data-root /scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/dataset/prepared_stream \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --baseline-ckpt "${GCV1_K3}" \
  --resume-ckpt "${RESUME_CKPT}" \
  --resume-step 30000 \
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
  --max-steps 40000 --eval-every 500 --checkpoint-every 1000 \
  --lr 5e-6 --weight-decay 1e-4 --precision bf16 \
  --temporal-backbone mamba \
  --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 1 \
  --temporal-stateful \
  --temporal-insert-count 2 \
  --residual-ar-feedback gated \
  --feedback-lambda-init 1.0 \
  --feedback-lambda-final 1.0 \
  --feedback-lambda-warmup-start-step 30000 \
  --feedback-lambda-warmup-end-step 30000 \
  --stop-grad-feedback \
  --grad-clip-norm 0.5 \
  --residual-output-head enabled \
  --trainable-part proc_and_mamba \
  --memory-mode optimal \
  --seed 22

echo "B-route ablation done: λ=1.0 plateau 10k from fast Stage C step30000, lr=5e-6"
