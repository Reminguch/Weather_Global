#!/usr/bin/env bash
# Batch eval for all v15 v4 ckpts (1k-17k) K=1 single-step.

#SBATCH --job-name=v15v4-eval-all
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v15v4_eval_all_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v15v4_eval_all_%j.err

set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments
CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v15_v4_v13arch_7yr_dstate32/v15_v4_20k_dstate32
OUT_DIR=${CKPT_BASE}/eval_results
mkdir -p "${OUT_DIR}"

for STEP in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 12000 13000 14000 15000 16000; do
  CKPT="${CKPT_BASE}/v13_residual_step${STEP}.pkl"
  OUT="${OUT_DIR}/eval_step${STEP}.json"
  [[ -f "${OUT}" ]] && { echo "[skip] step ${STEP}"; continue; }
  [[ ! -f "${CKPT}" ]] && { echo "[skip] no ckpt step ${STEP}"; continue; }
  echo "===== eval step ${STEP} ====="
  ${PYTHON} -u scripts/training/full_mamba_v9/eval_v9.py \
    --ckpt "${CKPT}" \
    --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
    --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
    --residual-msg-steps 2 \
    --temporal-location mesh_processor_interleaved \
    --temporal-hidden-size 128 --temporal-d-state 32 --temporal-d-conv 4 \
    --temporal-dt-rank auto --temporal-layers 2 \
    --n-samples 32 \
    --out-json "${OUT}"
done
echo "[done]"
