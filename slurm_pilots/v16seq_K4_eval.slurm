#!/usr/bin/env bash
# K=12 rollout eval for v16 seq curriculum Stage 2 K=2 (step 23000).
#SBATCH --job-name=v16seq-K4-eval
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v16seq_K4_eval_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v16seq_K4_eval_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v9/eval_v9_rollout.py
cd /home/lm8598/Weather_Global_experiments

CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v16_seq_curriculum/stage3_K4
EVAL_OUT_DIR=${CKPT_BASE}/eval_results
mkdir -p "${EVAL_OUT_DIR}"

${PYTHON} -u ${SCRIPT} \
  --ckpt "${CKPT_BASE}/v13_residual_step26000.pkl" \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 12 \
  --n-samples 32 \
  --out-json "${EVAL_OUT_DIR}/eval_step26000_rollout_K12.json"
