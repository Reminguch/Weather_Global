#!/usr/bin/env bash
# Warmup-duration sweep on K=12 best ckpt (res=1).
# W ∈ {24, 48, 72, 96, 120} steps × 6h = {6, 12, 18, 24, 30} days truth warmup.
# All tasks use --align-w-max=120 so AR phase starts at the SAME calendar time
# across W's (apples-to-apples comparison of warmup duration only).
#SBATCH --job-name=eval-K12-W-sweep
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --constraint=gpu80
#SBATCH --time=02:30:00
#SBATCH --array=0-4
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/eval_K12_Wsweep_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/eval_K12_Wsweep_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

W_LIST=(24 48 72 96 120)
W=${W_LIST[$SLURM_ARRAY_TASK_ID]}
W_MAX=120

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
BASELINE_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
RUN_DIR=${ROOT}/results/gcmamba_v2/gcmamba_v2_K12_from_K1step18k
GCMAMBA_CKPT=$(ls -1t "${RUN_DIR}"/ckpt_step*.npz | head -1)
STEP=$(basename "${GCMAMBA_CKPT}" | sed -nE 's/ckpt_step([0-9]+).*\.npz/\1/p')

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-06-16-gcmamba_v2
mkdir -p "${OUT_DIR}"
OUT_JSON="${OUT_DIR}/eval_res1_2way_K12_step${STEP}_W${W}.json"

echo "================================================"
echo "=== task_id=${SLURM_ARRAY_TASK_ID}  W=${W} (alignWmax=${W_MAX})  ckpt=${GCMAMBA_CKPT}"
echo "================================================"

$PYTHON -u scripts/training/gcmamba_v2_diagnostics/eval_K1_2way.py \
  --baseline-ckpt "${BASELINE_CKPT}" \
  --gc-mamba-ckpt "${GCMAMBA_CKPT}" \
  --resolution 1.0 --mesh-size 5 --width 512 --processor-msg-steps 16 \
  --temporal-location mesh_post_processor \
  --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-layers 1 --temporal-insert-count 1 \
  --n-anchors 32 --warmup-steps ${W} --target-steps 40 --align-w-max ${W_MAX} \
  --seed 0 \
  --out-json "${OUT_JSON}"

echo "W=${W} eval done — ${OUT_JSON}"
