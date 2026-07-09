#!/usr/bin/env bash
#SBATCH --job-name=v12bm-eval
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v12bm_eval_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v12bm_eval_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v11/eval_v11.py
cd /home/lm8598/Weather_Global_experiments

CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v12_fast/v12_bigmamba_20k
EVAL_OUT_DIR=${CKPT_BASE}/eval_results
mkdir -p "${EVAL_OUT_DIR}"

# Eval at key milestones: matching v12's best (3000) + 10K + 20K
for STEP in 1000 3000 5000 10000 20000; do
  CKPT="${CKPT_BASE}/v12_residual_step${STEP}.pkl"
  [[ ! -f "${CKPT}" ]] && { echo "skip ${STEP}"; continue; }
  echo "============ EVAL bigmamba step ${STEP} ============"
  ${PYTHON} -u ${SCRIPT} \
    --ckpt "${CKPT}" \
    --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
    --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
    --temporal-d-inner 256 \
    --temporal-d-state 16 --temporal-d-conv 4 \
    --temporal-dt-rank 256 --temporal-layers 2 \
    --n-samples 32 \
    --out-json "${EVAL_OUT_DIR}/eval_step${STEP}.json"
done
