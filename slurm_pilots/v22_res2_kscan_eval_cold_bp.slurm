#!/usr/bin/env bash
# v22 res=2 K-scan cold_bp eval — check if eval RMSE decays with training step
# Array: 6 K values × 3 ckpt-steps each = 18 tasks
#SBATCH --job-name=v22-res2-eval-kscan
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --array=0-17
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22_res2_eval_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22_res2_eval_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

# Task layout: (K, ckpt_step). Step picks span the training trajectory.
TASK_K=(    2    2    2    4    4    4    8    8    8    10   10   10   14   14   14   18   18   18)
TASK_STEP=( 500  2000 5000 500  2000 5000 500  2000 4000 500  2000 3000 500  1500 3000 500  1000 2000)

K=${TASK_K[$SLURM_ARRAY_TASK_ID]}
STEP=${TASK_STEP[$SLURM_ARRAY_TASK_ID]}

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
CKPT=${ROOT}/results/v22_res2/K${K}_res2_fresh/v13_residual_step${STEP}.pkl
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-25-v22-res2-eval-kscan
mkdir -p ${OUT_DIR}
OUT_JSON=${OUT_DIR}/K${K}_step${STEP}_cold_bp.json

[ -f "${CKPT}" ] || { echo "ERR: missing ckpt ${CKPT}"; exit 1; }

echo "================================================"
echo "=== K=${K} step=${STEP} cold_bp eval"
echo "=== ckpt=${CKPT}"
echo "================================================"

$PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt ${CKPT} \
  --ckpt-in ${GCV1_K3} \
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --eval-mode cold_bp \
  --target-steps 40 \
  --warmup-steps 24 \
  --n-samples 32 \
  --seed 0 \
  --out-json "${OUT_JSON}"

echo "K=${K} step=${STEP} cold_bp done"
