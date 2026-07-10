#!/usr/bin/env bash
# REGRESSION TEST (res=1): retrain v22cl K=18 fresh-from-DM for 1000 steps from the
# frozen CLEAN snapshot, same seed/base/data as the original run. Compare:
#   1. training-loss trajectory vs original train_log.json
#   2. (optional) step-1000 ckpt eval vs original step-1000
# Original reference: /scratch/.../results/v22_res1_closedloop_fresh/K18_closedloop_sg_fresh_dmckpt/
#SBATCH --job-name=test-v22cleanfrozen-r1-train
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/test_v22cleanfrozen_r1_train_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/test_v22cleanfrozen_r1_train_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_v22clean_frozen   # FROZEN CLEAN tree; relative script path

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=${ROOT}/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RESIDUAL_ROOT=${ROOT}/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2
[ -f "${DM_CKPT}" ] || { echo "ERR: missing DM GC-small ckpt"; exit 1; }

OUT=${ROOT}/results/v22_res1_closedloop_frozen_test
mkdir -p ${OUT}

echo "=== TEST: frozen-clean res=1 K=18 fresh-from-DM 0->1000, seed 22 ==="
${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${DM_CKPT}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K18_closedloop_sg_fresh_dmckpt_frozentest \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 1000 --checkpoint-every 500 \
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

echo "=== TEST res=1 training done ==="
