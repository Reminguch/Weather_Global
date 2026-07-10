#!/usr/bin/env bash
# v22 K=22 city trace including MSLP, all 4 seasonal anchors.
#SBATCH --job-name=v22-K22-city-mslp
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22_K22_city_mslp_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22_K22_city_mslp_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ANCHORS=("2022-01-15T00:00:00" "2022-04-15T00:00:00" "2022-07-15T00:00:00" "2022-10-15T00:00:00")
TAGS=("20220115" "20220415" "20220715" "20221015")
ANCHOR=${ANCHORS[$SLURM_ARRAY_TASK_ID]}
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-20-v22-city-mslp
mkdir -p ${OUT_DIR}

$PYTHON -u scripts/training/full_mamba_v23/eval_city_trace_mslp.py \
  --ckpt ${ROOT}/results/v22/K22_from_v22K1_23k/v13_residual_step26000.pkl \
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 40 \
  --anchor-date "${ANCHOR}" \
  --out-json ${OUT_DIR}/v22_K22_cities_mslp_${TAG}.json
