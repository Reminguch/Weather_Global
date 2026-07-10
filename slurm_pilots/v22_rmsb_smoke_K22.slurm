#!/usr/bin/env bash
# K=22 warm_full RMSB smoke test in v22clean worktree (= AR-Training-Lianghong branch).
# Goal: verify baseline RMSE doesn't blow up (lead 1≈0.475, lead 2≈0.54).
# n_samples=8 keeps it fast (~10 min).
#SBATCH --job-name=v22-rmsb-smoke-K22
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22_rmsb_smoke_K22_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22_rmsb_smoke_K22_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
# CRITICAL: cd into v22clean worktree so the script picks up clean graphcast.py
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT_DATA=/scratch/gpfs/DABANIN/lm8598/Weather_Global
CKPT=${ROOT_DATA}/results/v22/K22_from_v22K1_23k/v13_residual_step26000.pkl

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-19-v22-rmsb-rerun
mkdir -p ${OUT_DIR}
OUT_JSON=${OUT_DIR}/v22_K22_warm_full_smoke.json

echo "================================================"
echo "=== K=22 warm_full RMSB smoke (v22clean worktree at AR-Training-Lianghong)"
echo "=== ckpt=${CKPT}"
echo "=== pwd=$(pwd)"
echo "=== graphcast.py md5=$(md5sum third_party/graphcast/graphcast/graphcast.py | awk '{print $1}')"
echo "================================================"

$PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt ${CKPT} \
  --data-path ${ROOT_DATA}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir ${ROOT_DATA}/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --eval-mode warm_full \
  --target-steps 40 \
  --warmup-steps 24 \
  --n-samples 8 \
  --seed 0 \
  --out-json "${OUT_JSON}"

echo "K=22 warm_full RMSB smoke done"
