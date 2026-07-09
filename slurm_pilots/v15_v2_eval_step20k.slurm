#!/usr/bin/env bash
# v15 v2 step 20000 (final) eval — K=1 and K=6 rollout, on NEW WB2 2022.

#SBATCH --job-name=v15v2-eval-20k
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v15v2_eval_20k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v15v2_eval_20k_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v15_v13arch_7yr_v2/v15_20k_v2
EVAL_OUT_DIR=${CKPT_BASE}/eval_results
mkdir -p "${EVAL_OUT_DIR}"

CKPT="${CKPT_BASE}/v13_residual_step20000.pkl"
DATA=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
STATS=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats
COMMON="--ckpt ${CKPT} --data-path ${DATA} --stats-dir ${STATS} \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --n-samples 32"

echo "============= K=1 (6h lead) ============="
${PYTHON} -u /home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v9/eval_v9.py \
  ${COMMON} --out-json "${EVAL_OUT_DIR}/eval_step20000.json"

echo
echo "============= K=6 rollout (6h..36h) ============="
${PYTHON} -u /home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v9/eval_v9_rollout.py \
  ${COMMON} --target-steps 6 --out-json "${EVAL_OUT_DIR}/eval_step20000_rollout_K6.json"
