#!/usr/bin/env bash
#SBATCH --job-name=v9-diag
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v9_diag_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v9_diag_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v9/diagnose_v9.py
cd /home/lm8598/Weather_Global_experiments

CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v9_asym_residual_mamba/v9_arm_10k

for STEP in 500 2000 5000 9000; do
  CKPT="${CKPT_BASE}/v9_residual_step${STEP}.pkl"
  if [[ ! -f "${CKPT}" ]]; then
    echo "MISSING ${CKPT}, skipping"
    continue
  fi
  echo "=========================================="
  echo "DIAGNOSTIC for ckpt step ${STEP}"
  echo "=========================================="
  ${PYTHON} -u ${SCRIPT} --ckpt "${CKPT}" --n-samples 16
  echo
done
