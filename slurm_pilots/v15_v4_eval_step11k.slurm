#!/usr/bin/env bash
# Eval v15 v4 step 11000 (~epoch 8.6) — K=1 + K=6 rollout RMSE/MAE.

#SBATCH --job-name=v15v4-eval-11k
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v15v4_eval_11k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v15v4_eval_11k_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v15_v4_v13arch_7yr_dstate32/v15_v4_20k_dstate32
EVAL_OUT_DIR=${CKPT_BASE}/eval_results
mkdir -p "${EVAL_OUT_DIR}"

CKPT="${CKPT_BASE}/v13_residual_step11000.pkl"
COMMON="--ckpt ${CKPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 32 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --n-samples 32"

echo "============= K=1 ============="
${PYTHON} -u scripts/training/full_mamba_v9/eval_v9.py ${COMMON} --out-json "${EVAL_OUT_DIR}/eval_step11000.json"

echo
echo "============= K=6 rollout ============="
${PYTHON} -u scripts/training/full_mamba_v9/eval_v9_rollout.py ${COMMON} \
  --target-steps 6 --out-json "${EVAL_OUT_DIR}/eval_step11000_rollout_K6.json"
