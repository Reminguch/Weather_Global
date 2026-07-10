#!/usr/bin/env bash
# Residual-alpha sweep for K=22 at TWO ckpts (step 6000 peak vs step 20000 overfit),
# cold_full mode, 4 alpha values.
# 2 ckpts × 4 alphas = 8 tasks.
#
# If step 20k with alpha<1 approaches step 6k peak, correction-amplitude overfit
# is confirmed.
#SBATCH --job-name=v22cl-K22-alpha
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --array=0-7
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_K22_alpha_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_K22_alpha_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
DATA=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-07-02-v22cl-K22-alpha-sweep
mkdir -p ${OUT_DIR}

# 2 ckpts × 4 alphas
STEPS=(6000 6000 6000 6000 20000 20000 20000 20000)
ALPHAS=(0.25 0.5 0.75 1.0 0.25 0.5 0.75 1.0)
STEP=${STEPS[$SLURM_ARRAY_TASK_ID]}
ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}

CKPT=${ROOT}/results/v22_res1_closedloop_fresh/K22_closedloop_sg_fresh_dmckpt/v13_residual_step${STEP}.pkl
ALPHA_TAG=$(echo "${ALPHA}" | tr '.' 'p')
OUT_JSON=${OUT_DIR}/v22cl_r1_K22_step${STEP}_cold_full_alpha${ALPHA_TAG}_zero.json

echo "=== K=22 step=${STEP} cold_full alpha=${ALPHA} ==="
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
  --residual-alpha ${ALPHA} \
  --out-json "${OUT_JSON}"
echo "Done step=${STEP} alpha=${ALPHA}"
