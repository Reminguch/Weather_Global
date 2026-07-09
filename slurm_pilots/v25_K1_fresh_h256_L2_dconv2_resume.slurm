#!/usr/bin/env bash
# v25 K=1 fresh — RESUME job (afterany).
#SBATCH --job-name=v25-K1-fresh-resume
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=36:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v25_fresh_resume_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v25_fresh_resume_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_CPP_MIN_LOG_LEVEL=1
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v24/train_mz_v24.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RESIDUAL_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v25
RUN_DIR=${OUT}/K1_fresh_h256_L2_d2_23k
TARGET_STEP=23000

if [ ! -d "${RUN_DIR}" ]; then
  echo "ERROR: run dir does not exist: ${RUN_DIR}" 1>&2
  exit 1
fi

LATEST_CKPT=$(ls -1 "${RUN_DIR}"/v13_residual_step*.pkl 2>/dev/null | sort -V | tail -1 || true)
if [ -z "${LATEST_CKPT}" ]; then
  echo "ERROR: no ckpts found in ${RUN_DIR}" 1>&2
  exit 1
fi

LATEST_STEP=$(basename "${LATEST_CKPT}" | sed 's/.*_step\([0-9]*\)\.pkl/\1/')
echo "Found latest ckpt: ${LATEST_CKPT}"
echo "Latest step: ${LATEST_STEP}, target: ${TARGET_STEP}"

if [ "${LATEST_STEP}" -ge "${TARGET_STEP}" ]; then
  echo "Already at step ${LATEST_STEP} >= ${TARGET_STEP}. Nothing to do, exiting 0."
  exit 0
fi

START_STEP=$((LATEST_STEP + 1))
echo "Resuming from step ${START_STEP} to ${TARGET_STEP}"

${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K1_fresh_h256_L2_d2_23k \
  --resume-from "${LATEST_CKPT}" \
  --start-step "${START_STEP}" \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps ${TARGET_STEP} --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 25 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 24 --bptt-steps 24 \
  --ar-tail-K 0 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 256 \
  --temporal-d-state 16 --temporal-d-conv 2 --temporal-dt-rank auto \
  --temporal-layers 2
