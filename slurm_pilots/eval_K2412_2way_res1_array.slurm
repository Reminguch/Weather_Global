#!/usr/bin/env bash
# Multi-K city eval at res=1 (DeepMind small backbone):
# Compares DeepMind baseline vs gc_mamba_K=K (latest available ckpt) for K=2,4,8,12.
# 32 anchors × (24 truth warmup + 40 closed-loop rollout). City temp + precip vs ERA5.
#SBATCH --job-name=eval-gcm-res1-K2412
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_gcm_res1_K2412_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_gcm_res1_K2412_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

K_LIST=(2 4 8 12)
K=${K_LIST[$SLURM_ARRAY_TASK_ID]}

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
BASELINE_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
RUN_DIR=${ROOT}/results/gcmamba_v2/gcmamba_v2_K${K}_from_K1step18k
GCMAMBA_CKPT=$(ls -1t "${RUN_DIR}"/ckpt_step*.npz | head -1)
STEP=$(echo "${GCMAMBA_CKPT}" | grep -oE 'step[0-9]+' | sed 's/step//')

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-16-gcmamba_v2
mkdir -p "${OUT_DIR}"
OUT_JSON="${OUT_DIR}/eval_res1_2way_K${K}_step${STEP}.json"

echo "================================================"
echo "=== K=${K}  ckpt=${GCMAMBA_CKPT}  step=${STEP}"
echo "================================================"

$PYTHON -u scripts/training/gcmamba_v2_diagnostics/eval_K1_2way.py \
  --baseline-ckpt "${BASELINE_CKPT}" \
  --gc-mamba-ckpt "${GCMAMBA_CKPT}" \
  --resolution 1.0 --mesh-size 5 --width 512 --processor-msg-steps 16 \
  --temporal-location mesh_post_processor \
  --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-layers 1 --temporal-insert-count 1 \
  --n-anchors 32 --warmup-steps 24 --target-steps 40 \
  --seed 0 \
  --out-json "${OUT_JSON}"

echo "K=${K} eval done — ${OUT_JSON}"
