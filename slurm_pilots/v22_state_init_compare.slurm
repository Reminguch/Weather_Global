#!/usr/bin/env bash
# Compare 3 residual_state initialization modes for K=18 step 20000:
#   1. cold_bp + init=ckpt   (current default)
#   2. cold_bp + init=zero   (clean SSM start)
#   3. warm_bp               (24 truth-fed warmup before cold rollout)
#SBATCH --job-name=v22-state-init-cmp
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22_state_init_cmp_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22_state_init_cmp_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-6-25-Res2Mamba

CKPT=${ROOT}/results/v22_res2/K18_res2_fresh/v13_residual_step20000.pkl

COMMON_ARGS="
  --ckpt ${CKPT}
  --ckpt-in ${GCV1_K3}
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats
  --resolution 2.0 --mesh-size 4 --width 512
  --baseline-msg-steps 6 --residual-msg-steps 2
  --val-year 2022 --train-start-year 2015 --train-end-year 2021
  --input-duration 12h
  --temporal-location mesh_processor_interleaved
  --temporal-hidden-size 128 --temporal-d-inner 128
  --temporal-d-state 16 --temporal-d-conv 4
  --temporal-dt-rank auto --temporal-layers 2
  --target-steps 40 --warmup-steps 24
  --n-samples 32 --seed 0"

# Run 1: cold_bp with init=zero
echo "=== K=18 cold_bp init=ZERO ==="
$PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py $COMMON_ARGS \
  --eval-mode cold_bp --residual-state-init zero \
  --out-json ${OUT_DIR}/K18_step20000_cold_bp_init_zero.json

# Run 2: warm_bp (warm24 init, baseline rollout)
echo "=== K=18 warm_bp (warm24 init) ==="
$PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py $COMMON_ARGS \
  --eval-mode warm_bp --residual-state-init ckpt \
  --out-json ${OUT_DIR}/K18_step20000_warm_bp.json

# Note: cold_bp init=ckpt is already in /K18_step20000_cold_bp.json

echo "All done"
