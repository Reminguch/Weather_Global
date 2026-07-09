#!/usr/bin/env bash
# v22 K-sweep RMSB eval: cold_bp (open-loop deployment) + warm_full (closed-loop deployment)
# for K=1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22.
# Uses patched eval_v22_clean.py that accumulates per-cell bias and writes
# rmsb_baseline/rmsb_full per (variable, lead) into JSON.
#
# 12 K × 2 modes = 24 evals; ~30 min each on GPU → ~12 h total.
#SBATCH --job-name=v22-rmsb-kscan
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=20:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22_rmsb_kscan_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22_rmsb_kscan_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-15-v22-rmsb-kscan
mkdir -p ${OUT_DIR}

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global

# Map K → ckpt path
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

K_LIST="1 2 4 6 8 10 12 14 16 18 20 22"
MODES="cold_bp warm_full"

for MODE in $MODES; do
  for K in $K_LIST; do
    CKPT=${CKPTS[$K]}
    OUT_JSON=${OUT_DIR}/v22_K${K}_${MODE}.json
    if [ -f "${OUT_JSON}" ]; then
      echo "skip (exists): ${OUT_JSON}"
      continue
    fi
    echo "================================================"
    echo "=== K=${K} mode=${MODE}  ckpt=${CKPT}"
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
  done
done

echo "v22 RMSB K-sweep done"
