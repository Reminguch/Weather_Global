#!/usr/bin/env bash
# Eval sweep for K=22 cold_full at intermediate ckpts every 2000 steps.
# Steps 14000 and 20000 already have eval JSONs — this fills in 6, 8, 10, 12, 16, 18.
# 6 tasks × ~35 min each; will run in parallel across biao GPUs.
#SBATCH --job-name=v22cl-K22-sweep
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --array=0-5
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_r1_K22_sweep_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_r1_K22_sweep_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-07-02-v22cl-r1-K22-sweep
mkdir -p ${OUT_DIR}

# 6 intermediate ckpts to fill in (14000 and 20000 already evaluated separately)
STEPS=(6000 8000 10000 12000 16000 18000)
STEP=${STEPS[$SLURM_ARRAY_TASK_ID]}
K=22
MODE=cold_full

CKPT=${ROOT}/results/v22_res1_closedloop_fresh/K${K}_closedloop_sg_fresh_dmckpt/v13_residual_step${STEP}.pkl
OUT_JSON=${OUT_DIR}/v22cl_r1_K${K}_step${STEP}_${MODE}_zero.json

[ -f "${CKPT}" ] || { echo "[skip] ${CKPT} not present"; exit 0; }

echo "=== v22cl K=${K} step ${STEP} ${MODE} eval ==="
$PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt ${CKPT} --ckpt-in "${DM_CKPT}" \
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --eval-mode ${MODE} --residual-state-init zero \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
  --out-json "${OUT_JSON}"
echo "Done step=${STEP}"
