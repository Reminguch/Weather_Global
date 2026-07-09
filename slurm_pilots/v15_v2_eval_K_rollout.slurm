#!/usr/bin/env bash
# Autoregressive K-step rollout eval for v15 v2 step 17000 ckpt.
# K=2 (12h), K=4 (24h), K=6 (36h) lead times.
# GraphCast's predict_K_steps_unrolled handles rollout internally; mamba
# stateful carries through the unroll.

#SBATCH --job-name=v15v2-eval-K
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v15v2_eval_K_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v15v2_eval_K_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v9/eval_v9.py
cd /home/lm8598/Weather_Global_experiments

CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v15_v13arch_7yr_v2/v15_20k_v2
EVAL_OUT_DIR=${CKPT_BASE}/eval_results
mkdir -p "${EVAL_OUT_DIR}"

CKPT="${CKPT_BASE}/v13_residual_step17000.pkl"

for K in 2 4 6; do
  OUT="${EVAL_OUT_DIR}/eval_step17000_K${K}.json"
  echo "============= K=${K} (lead $((K*6))h) ============="
  ${PYTHON} -u ${SCRIPT} \
    --ckpt "${CKPT}" \
    --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
    --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
    --residual-msg-steps 2 \
    --temporal-location mesh_processor_interleaved \
    --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
    --temporal-dt-rank auto --temporal-layers 2 \
    --target-steps ${K} \
    --n-samples 32 \
    --out-json "${OUT}"
done
