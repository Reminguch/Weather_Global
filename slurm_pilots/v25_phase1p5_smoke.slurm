#!/usr/bin/env bash
# v25 phase 1.5 SMOKE: unlock test for gated extreme head.
# Changes from phase 1:
#   --gate-bias-init -1 (sigmoid ≈ 0.27, not 0.029) → r_ext gradient ~9x less throttled
#   --gate-beta 0.2 (4x stronger BCE on gate)
#   --gate-alpha-extreme 2.0 (extreme-weighted MSE → ext_head value-learning signal)
#   --ext-head-lr 1e-4 (33x base LR; fresh-init head needs fast learning)
#   --gate-head-lr 3e-5 (10x base LR)
#   --lr 3e-6 (kept low for backbone but base remains FROZEN via --gate-phase 1)
#   --max-steps 5000 (vs phase 1: 3000)
# Init from K=18 res=2 step20000, freeze base/Mamba, train only ext+gate heads.
#SBATCH --job-name=v25-phase1p5
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=10:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v25_phase1p5_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v25_phase1p5_%j.err
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
OUT=${ROOT}/results/v25_gated_extreme

mkdir -p ${OUT}
[ -f "${INIT_CKPT}" ] || { echo "ERR: missing K=18 init ckpt"; exit 1; }

${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${GCV1_K3}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K18_gext_phase1p5_smoke \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 5000 --checkpoint-every 500 \
  --lr 3e-6 --weight-decay 0.0 --seed 25 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 100 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 18 \
  --feedback-mode baseline \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2 \
  --gated-extreme-head \
  --init-from-residual-ckpt "${INIT_CKPT}" \
  --gate-phase 1 \
  --gate-bias-init -1.0 \
  --gate-beta 0.2 \
  --gate-lambda-g 0.001 \
  --gate-extreme-quantile 0.95 \
  --gate-alpha-extreme 2.0 \
  --ext-head-lr 1e-4 \
  --gate-head-lr 3e-5

echo "v25 phase 1.5 smoke done"
