#!/usr/bin/env bash
# Full 4-mode eval for v22 closed-loop K=18 res=2 ckpts at step 500/1000/1500/2000.
# Modes: cold_bp / cold_full / warm_bp / warm_full (all --residual-state-init zero).
# Array: 16 tasks = 4 steps × 4 modes.
#SBATCH --job-name=v22cl-eval-4mode
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=02:30:00
#SBATCH --array=0-15%6
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_eval_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_eval_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
RUN_DIR=${ROOT}/results/v22_res2_closedloop/K18_res2_closedloop_sg_fresh
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-6-27-v22cl-K18-eval
mkdir -p ${OUT_DIR}

STEPS=(500 1000 1500 2000)
MODES=(cold_bp cold_full warm_bp warm_full)

STEP_IDX=$((SLURM_ARRAY_TASK_ID / 4))
MODE_IDX=$((SLURM_ARRAY_TASK_ID % 4))
STEP=${STEPS[$STEP_IDX]}
MODE=${MODES[$MODE_IDX]}

CKPT=${RUN_DIR}/v13_residual_step${STEP}.pkl
OUT_JSON=${OUT_DIR}/K18cl_step${STEP}_${MODE}_zero.json

[ -f "${CKPT}" ] || { echo "[skip] ${CKPT} not present yet"; exit 0; }

echo "================================================"
echo "=== v22cl K=18 step${STEP}, mode=${MODE}, zero-init"
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
  --eval-mode ${MODE} \
  --residual-state-init zero \
  --target-steps 40 \
  --warmup-steps 24 \
  --n-samples 32 \
  --seed 0 \
  --out-json "${OUT_JSON}"

echo "v22cl K=18 step${STEP} ${MODE} done"
