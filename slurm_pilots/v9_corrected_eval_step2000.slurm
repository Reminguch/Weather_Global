#!/usr/bin/env bash
#SBATCH --job-name=v9-corr-eval
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v9_corr_eval_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v9_corr_eval_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v9/eval_v9.py
cd /home/lm8598/Weather_Global_experiments

CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v9_asym_residual_mamba/v9_full_train_10k
EVAL_OUT_DIR=${CKPT_BASE}/eval_results
mkdir -p "${EVAL_OUT_DIR}"

for STEP in 1000 2000; do
  CKPT="${CKPT_BASE}/v9_residual_step${STEP}.pkl"
  if [[ ! -f "${CKPT}" ]]; then
    echo "MISSING ${CKPT}, skipping"
    continue
  fi
  echo "============ EVAL v9-corrected step ${STEP} ============"
  ${PYTHON} -u ${SCRIPT} \
    --ckpt "${CKPT}" \
    --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
    --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
    --residual-msg-steps 2 \
    --temporal-location mesh_processor_interleaved \
    --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
    --temporal-dt-rank auto --temporal-layers 2 \
    --n-samples 32 \
    --out-json "${EVAL_OUT_DIR}/eval_step${STEP}.json"
done
