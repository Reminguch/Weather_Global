#!/usr/bin/env bash
# v30a SMOKE B0-v5 — middle-ground LR probe.
#
# B0-v3 (unnorm): stuck at L_con=log(5), grad_norm ~1e-3 → embedding collapse
#                 masked by scale dominance of raw K/Pa values.
# B0-v4 (norm+lr=1e-3): unblocked but chaotic — L_con oscillates 0.5–5.8,
#                       grad_norm 0.1–2.2, gap jumps ±0.4 per step. LR too high.
#
# B0-v5 changes:
#   event_head_lr           1e-3 → 1e-4   (10x smaller, effective ~1e-4)
#   event_target_enc_lr     1e-3 → 1e-4
#   lambda_con              stays 1.0
#   grad_clip                1.0 → 0.5     (mild tightening)
#   steps                    300 → 500     (need more steps at lower LR)
# Everything else held constant (sampler, T=1, patch=16, Nneg=4, tau=0.1).
#
# Success criteria (via post-run EMA):
#   L_con_EMA_50 no longer 0.5–5.8; monotonic-ish or narrow band
#   gap_EMA_50 clearly > 0
#   top1_EMA_50 > 0.4  (random baseline 1/5 = 0.20)
#   grad_norm_EMA_50 stable, same order of magnitude across 100+ steps
#
#SBATCH --job-name=v30a-B0v5
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v30a_B0v5_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v30a_B0v5_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
INIT_CKPT=${ROOT}/results/v22_res2/K18_res2_fresh/v13_residual_step20000.pkl
PREPARED_ROOT=/scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/dataset/prepared_stream/res2
RESIDUAL_ROOT=${ROOT}/data/graphcast/graphcast/dataset/precomputed_residuals/v22_res2_GCv1K3_synthetic
OUT=${ROOT}/results/v30a_B0

mkdir -p ${OUT}
[ -f "${INIT_CKPT}" ] || { echo "ERR: missing K=18 init ckpt"; exit 1; }

echo "=== v30a B0-v5: MIDDLE-LR probe (lambda_con=1.0, event_lr=1e-4, grad_clip=0.5, 500 steps) ==="

${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${GCV1_K3}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name B0v5_mid_lr_probe \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 500 --checkpoint-every 100 \
  --lr 3e-6 --weight-decay 0.0 --seed 30 --precision bf16 \
  --grad-clip 0.5 --warmup-steps 0 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 18 \
  --feedback-mode baseline \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2 \
  --event-adapter-head \
  --init-from-residual-ckpt "${INIT_CKPT}" \
  --gate-phase 1 \
  --adapter-scale 0.0 \
  --lambda-con 1.0 \
  --lambda-delta-reg 0.0 \
  --tau-contrastive 0.1 \
  --event-patch-size 16 \
  --event-t-win 1 \
  --event-c-vars 4 \
  --event-trunk-dim 64 --event-proj-dim 64 \
  --event-gate-bias-init -2.0 --damp-bias-init -4.0 --event-delta-cap 0.5 \
  --event-head-lr 1e-4 \
  --event-target-encoder-lr 1e-4 \
  --contrastive-anchor-idx 23 \
  --n-neg 4 \
  --event-min-time-gap-days 7

echo "=== B0-v5 done ==="
