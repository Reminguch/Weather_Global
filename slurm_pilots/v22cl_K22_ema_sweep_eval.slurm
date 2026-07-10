#!/usr/bin/env bash
# Eval K=22 EMA variants (different beta) + SWA-all vs original SWA 5k-10k.
# 5 new ckpts × cold_full = 5 tasks.
#SBATCH --job-name=v22cl-K22-ema-sweep
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --array=0-4
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_K22_ema_sweep_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_K22_ema_sweep_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
DATA=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
EMA_DIR=${ROOT}/results/v22cl_r1_EMA
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-07-02-v22cl-K22-EMA-sweep
mkdir -p ${OUT_DIR}

CKPTS=(
  "${EMA_DIR}/K22_ema_beta50.pkl"
  "${EMA_DIR}/K22_ema_beta80.pkl"
  "${EMA_DIR}/K22_ema_beta90.pkl"
  "${EMA_DIR}/K22_ema_beta95.pkl"
  "${EMA_DIR}/K22_swa_all.pkl"
)
TAGS=(ema50 ema80 ema90 ema95 swa_all)
CKPT=${CKPTS[$SLURM_ARRAY_TASK_ID]}
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}

OUT_JSON=${OUT_DIR}/v22cl_r1_K22_${TAG}_cold_full_zero.json
[ -f "${CKPT}" ] || { echo "[skip] ${CKPT} not present"; exit 0; }

echo "=== K=22 ${TAG} cold_full eval ==="
$PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt ${CKPT} --ckpt-in "${DM_CKPT}" \
  --data-path ${DATA} \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --eval-mode cold_full --residual-state-init zero \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
  --out-json "${OUT_JSON}"
echo "Done ${TAG}"
