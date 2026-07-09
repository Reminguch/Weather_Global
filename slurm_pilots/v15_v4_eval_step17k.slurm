#!/usr/bin/env bash
#SBATCH --job-name=v15v4-eval-17k
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v15v4_eval_17k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v15v4_eval_17k_%j.err

set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v15_v4_v13arch_7yr_dstate32/v15_v4_20k_dstate32
CKPT="${CKPT_BASE}/v13_residual_step17000.pkl"
COMMON="--ckpt ${CKPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 32 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --n-samples 32"

echo "============= K=1 ============="
${PYTHON} -u scripts/training/full_mamba_v9/eval_v9.py ${COMMON} --out-json "${CKPT_BASE}/eval_results/eval_step17000.json"

echo
echo "============= K=6 rollout ============="
${PYTHON} -u scripts/training/full_mamba_v9/eval_v9_rollout.py ${COMMON} \
  --target-steps 6 --out-json "${CKPT_BASE}/eval_results/eval_step17000_rollout_K6.json"
