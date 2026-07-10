#!/usr/bin/env bash
# v22 framework (github AR-Training-Lianghong, clean) K=1 smoke on res=2 + GCv1 K=3 baseline.
# Verifies: v22 training loop compiles + runs end-to-end with our setup.
#SBATCH --job-name=v22-res2-K1-smoke
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22_res2_K1_smoke_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22_res2_K1_smoke_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

# Use OUR setup: res=2 prepared store + GCv1 K=3 baseline + synthetic residual_root (just anchor split)
ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
PREPARED_ROOT=/scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/dataset/prepared_stream/res2
RESIDUAL_ROOT=${ROOT}/data/graphcast/graphcast/dataset/precomputed_residuals/v22_res2_GCv1K3_synthetic
OUT=${ROOT}/results/v22_res2/smoke

mkdir -p ${OUT}

[ -f "${GCV1_K3}" ] || { echo "ERR: missing GCv1 K3 ckpt"; exit 1; }
[ -d "${PREPARED_ROOT}" ] || { echo "ERR: missing res=2 prepared store"; exit 1; }
[ -d "${RESIDUAL_ROOT}" ] || { echo "ERR: missing residual_root"; exit 1; }

${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${GCV1_K3}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K1_res2_smoke \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 1500 --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 0 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2

echo "v22 res=2 K=1 smoke done"
