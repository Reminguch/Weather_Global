#!/usr/bin/env bash
# v15 v2 step 20000 K=15 rollout eval (for cross-check vs v16).
# Same 32 anchors and target_steps=12 so we can plot RMSE-vs-lead-time
# (lead 6h..90h) side by side with v16.

#SBATCH --job-name=v18v2-K1-a2a
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v18v2_K1_a2a_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v18v2_K1_a2a_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v9/eval_v9_rollout.py
cd /home/lm8598/Weather_Global_experiments

CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v18_v2/K1_from_v15v2_20k
EVAL_OUT_DIR=${CKPT_BASE}/eval_results
mkdir -p "${EVAL_OUT_DIR}"

CKPT="${CKPT_BASE}/v13_residual_step23000.pkl"
OUT="${EVAL_OUT_DIR}/eval_step23000_rollout_K15_a2a.json"

echo "============= v15 v2 step 20000 K=15 rollout (lead 6h..90h) ============="
${PYTHON} -u ${SCRIPT} \
  --ckpt "${CKPT}" \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 15 \
  --n-samples 32 \
  --out-json "${OUT}"
