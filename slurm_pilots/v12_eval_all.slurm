#!/usr/bin/env bash
# Evaluate all v12 20K ckpts (1k..20k step) we don't have yet, on 32 val samples.
# Each ckpt eval ~2-3 min. 17 ckpts = ~50 min total.

#SBATCH --job-name=v12-eval-all
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v12_eval_all_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v12_eval_all_%j.err

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

for STEP in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000; do
  CKPT="${CKPT_BASE}/v12_residual_step${STEP}.pkl"
  OUT="${EVAL_OUT_DIR}/eval_step${STEP}.json"
  if [[ -f "${OUT}" ]]; then
    echo "[skip] eval_step${STEP} already exists"
    continue
  fi
  if [[ ! -f "${CKPT}" ]]; then
    echo "[skip] missing ckpt at step ${STEP}"
    continue
  fi
  echo "============= EVAL step ${STEP} ============="
  ${PYTHON} -u ${SCRIPT} \
    --ckpt "${CKPT}" \
    --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
    --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
    --n-samples 32 \
    --out-json "${OUT}"
done

echo "all done"
