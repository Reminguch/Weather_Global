#!/usr/bin/env bash
# v22 RMSB K-sweep — PARALLEL array version (replaces serial 9776583).
# 24 independent tasks: 12 K × 2 modes (cold_bp / warm_full).
# Each task runs one eval_v22_clean.py invocation (~30-50 min on GPU).
# Skips JSONs that already exist (so partial K=1 cold_bp + K=2 cold_bp from
# 9776583 will be reused).
#SBATCH --job-name=v22-rmsb-array
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --array=0-23
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22_rmsb_array_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22_rmsb_array_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

# Map task_id (0..23) → (K, MODE)
K_LIST=(1 2 4 6 8 10 12 14 16 18 20 22)
MODES=(cold_bp warm_full)
TASK_ID=${SLURM_ARRAY_TASK_ID}
K_IDX=$(( TASK_ID % 12 ))
MODE_IDX=$(( TASK_ID / 12 ))
K=${K_LIST[$K_IDX]}
MODE=${MODES[$MODE_IDX]}

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-17-v22-rmse-rerun
mkdir -p ${OUT_DIR}

# Map K → ckpt
declare -A CKPTS
CKPTS[1]=${ROOT}/results/v22/K1_from_v15v2_20k/v13_residual_step23000.pkl
CKPTS[2]=${ROOT}/results/v22/K2_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[4]=${ROOT}/results/v22/K4_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[6]=${ROOT}/results/v22/K6_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[8]=${ROOT}/results/v22/K8_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[10]=${ROOT}/results/v22/K10_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[12]=${ROOT}/results/v22/K12_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[14]=${ROOT}/results/v22/K14_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[16]=${ROOT}/results/v22/K16_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[18]=${ROOT}/results/v22/K18_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[20]=${ROOT}/results/v22/K20_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[22]=${ROOT}/results/v22/K22_from_v22K1_23k/v13_residual_step26000.pkl

CKPT=${CKPTS[$K]}
OUT_JSON=${OUT_DIR}/v22_K${K}_${MODE}.json

if [ -f "${OUT_JSON}" ]; then
    echo "skip (exists): ${OUT_JSON}"
    exit 0
fi

echo "================================================"
echo "=== task_id=${TASK_ID}  K=${K}  mode=${MODE}"
echo "=== ckpt=${CKPT}"
echo "================================================"

$PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt ${CKPT} \
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --eval-mode ${MODE} \
  --target-steps 40 \
  --warmup-steps 24 \
  --n-samples 32 \
  --seed 0 \
  --out-json ${OUT_JSON}

echo "task_id=${TASK_ID} (K=${K} ${MODE}) done"
