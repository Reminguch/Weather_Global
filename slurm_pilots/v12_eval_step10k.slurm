#!/usr/bin/env bash
#SBATCH --job-name=v12-eval-10k
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v12_eval_10k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v12_eval_10k_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v11/eval_v11.py
cd /home/lm8598/Weather_Global_experiments

CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v12_fast/v12_fast_20k
EVAL_OUT_DIR=${CKPT_BASE}/eval_results
mkdir -p "${EVAL_OUT_DIR}"

for STEP in 1000 5000 10000; do
  CKPT="${CKPT_BASE}/v12_residual_step${STEP}.pkl"
  if [[ ! -f "${CKPT}" ]]; then
    echo "MISSING ${CKPT}, skipping"
    continue
  fi
  echo "=========================================="
  echo "EVAL v12 ckpt step ${STEP}"
  echo "=========================================="
  ${PYTHON} -u ${SCRIPT} \
    --ckpt "${CKPT}" \
    --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
    --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
    --n-samples 32 \
    --out-json "${EVAL_OUT_DIR}/eval_step${STEP}.json"
  echo
done
