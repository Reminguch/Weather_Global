#!/usr/bin/env bash
# v26 Physical Tail Calibration Head — Phase 1 (freeze backbone + base_head).
# Init from v22 K=18 res=2 step 20000 ckpt.
# Train ONLY temporal_tail_calibration_head; 5000 steps.
#
# Loss:
#   L = weighted_mse(r_base + Delta_tail, residual_target)
#     + alpha_tail · E[m_truth · (Delta_tail - sg(target - r_base))²]
#     + lambda_quiet · E[(1 - m_phys) · Delta_tail²]
#
# Variables calibrated (v1): 2m_temperature (high+low) + 10m wind u/v.
#SBATCH --job-name=v26-tail-p1
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=10:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v26_tail_p1_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v26_tail_p1_%j.err
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
OUT=${ROOT}/results/v26_tail_calib

mkdir -p ${OUT}
[ -f "${INIT_CKPT}" ] || { echo "ERR: missing K=18 init ckpt"; exit 1; }

${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${GCV1_K3}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K18_tail_calib_phase1 \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 5000 --checkpoint-every 500 \
  --lr 3e-6 --weight-decay 0.0 --seed 26 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 100 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 18 \
  --feedback-mode baseline \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2 \
  --tail-calib-head \
  --init-from-residual-ckpt "${INIT_CKPT}" \
  --gate-phase 1 \
  --tail-alpha 1.0 \
  --tail-lambda-quiet 0.01 \
  --tail-cap-T 0.5 \
  --tail-cap-wind 0.5 \
  --tail-z95 1.64 \
  --tail-tau 0.5 \
  --tail-head-lr 1e-4

echo "v26 phase 1 done"
