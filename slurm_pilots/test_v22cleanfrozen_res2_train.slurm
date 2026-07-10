#!/usr/bin/env bash
# REGRESSION TEST: retrain res=2 inner128 K=18 for 2000 steps from the frozen CLEAN
# snapshot, same seed/base/data as the original run. Compare:
#   1. training-loss trajectory vs original K18 run log
#   2. step-2000 ckpt eval vs original step-2000 eval
# Original reference: /scratch/.../results/v22_res2_closedloop/K18_res2_inner128/ (0->2000 phase)
#SBATCH --job-name=test-v22cleanfrozen-train
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=03:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/test_v22cleanfrozen_train_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/test_v22cleanfrozen_train_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_v22clean_frozen   # FROZEN CLEAN tree; relative script path below

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
PREPARED_ROOT=/scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/dataset/prepared_stream/res2
RESIDUAL_ROOT=${ROOT}/data/graphcast/graphcast/dataset/precomputed_residuals/v22_res2_GCv1K3_synthetic
[ -f "${GCV1_K3}" ] || { echo "ERR: missing GCv1 K3 ckpt"; exit 1; }

OUT_PARENT=${ROOT}/results/v22_res2_closedloop_frozen_test
mkdir -p ${OUT_PARENT}

echo "=== TEST: frozen-clean res=2 inner128 K=18 fresh 0->2000, seed 22 ==="
${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${GCV1_K3}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT_PARENT}" \
  --run-name K18_res2_inner128_frozentest \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 2000 --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 18 \
  --feedback-mode closed_loop_sg \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2

echo "=== TEST training done ==="
