#!/usr/bin/env bash
# v22 RMSB K-scan, cold_bp mode only (open-loop residual eval).
# 12 tasks: K = 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22.
# cold_bp = W=0, residual = instantaneous correction on baseline trajectory
#         (residual NOT fed back to next-step input). This is the standard
#         "open-loop" eval that the May 23 plots actually used (under different name).
#SBATCH --job-name=v22-rmsb-coldbp-kscan
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22_rmsb_coldbp_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22_rmsb_coldbp_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

K_LIST=(1 2 4 6 8 10 12 14 16 18 20 22)
K=${K_LIST[$SLURM_ARRAY_TASK_ID]}

ROOT_DATA=/scratch/gpfs/DABANIN/lm8598/Weather_Global
# Map K → ckpt
declare -A CKPTS
CKPTS[1]=${ROOT_DATA}/results/v22/K1_from_v15v2_20k/v13_residual_step23000.pkl
CKPTS[2]=${ROOT_DATA}/results/v22/K2_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[4]=${ROOT_DATA}/results/v22/K4_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[6]=${ROOT_DATA}/results/v22/K6_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[8]=${ROOT_DATA}/results/v22/K8_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[10]=${ROOT_DATA}/results/v22/K10_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[12]=${ROOT_DATA}/results/v22/K12_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[14]=${ROOT_DATA}/results/v22/K14_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[16]=${ROOT_DATA}/results/v22/K16_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[18]=${ROOT_DATA}/results/v22/K18_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[20]=${ROOT_DATA}/results/v22/K20_from_v22K1_23k/v13_residual_step26000.pkl
CKPTS[22]=${ROOT_DATA}/results/v22/K22_from_v22K1_23k/v13_residual_step26000.pkl

CKPT=${CKPTS[$K]}
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-19-v22-rmsb-rerun
mkdir -p ${OUT_DIR}
OUT_JSON=${OUT_DIR}/v22_K${K}_cold_bp.json

echo "================================================"
echo "=== K=${K} cold_bp RMSB (v22clean worktree)"
echo "=== ckpt=${CKPT}  pwd=$(pwd)"
echo "================================================"

$PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt ${CKPT} \
  --data-path ${ROOT_DATA}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir ${ROOT_DATA}/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --eval-mode cold_bp \
  --target-steps 40 \
  --warmup-steps 24 \
  --n-samples 32 \
  --seed 0 \
  --out-json "${OUT_JSON}"

echo "K=${K} cold_bp done"
