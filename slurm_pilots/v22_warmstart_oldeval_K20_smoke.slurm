#!/usr/bin/env bash
# Smoke test: run the OLD eval_v22_warmstart.py on K=20 ckpt to verify the model is fine
# and confirm the new eval_v22_clean.py has a bug.
#SBATCH --job-name=v22-oldeval-K20-smoke
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22_oldeval_K20_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22_oldeval_K20_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
CKPT=${ROOT}/results/v22/K20_from_v22K1_23k/v13_residual_step26000.pkl
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-17-v22-rmse-rerun
mkdir -p ${OUT_DIR}
OUT_JSON=${OUT_DIR}/v22_K20_OLDEVAL_warmstart.json

$PYTHON -u scripts/training/full_mamba_v23/eval_v22_warmstart.py \
  --ckpt ${CKPT} \
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 40 \
  --warmup-steps 24 \
  --n-samples 32 \
  --seed 0 \
  --out-json ${OUT_JSON}

echo "OLD eval K=20 done"
