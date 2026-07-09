#!/usr/bin/env bash
# Eval v14 ckpts at key steps. v14 uses GCResidualWithSpecialistHeads —
# almost identical to v9's GCResidualWithZeroHead, but with 2 heads. Need
# a v14-specific eval since the model class differs at the head.

#SBATCH --job-name=v14-eval-key
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v14_eval_key_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v14_eval_key_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v14/eval_v14.py
cd /home/lm8598/Weather_Global_experiments

CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v14_specialist_heads/v14_20k
EVAL_OUT_DIR=${CKPT_BASE}/eval_results
mkdir -p "${EVAL_OUT_DIR}"

# Key milestones: 5K = v13 best, 9K = epoch ~25 (specialist heads should help here),
# 15K = late (where v13 had degradation)
for STEP in 5000 9000 15000; do
  CKPT="${CKPT_BASE}/v14_residual_step${STEP}.pkl"
  [[ ! -f "${CKPT}" ]] && { echo "skip ${STEP}"; continue; }
  OUT="${EVAL_OUT_DIR}/eval_step${STEP}.json"
  echo "============= EVAL v14 step ${STEP} ============="
  ${PYTHON} -u ${SCRIPT} \
    --ckpt "${CKPT}" \
    --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
    --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
    --residual-msg-steps 2 \
    --temporal-location mesh_processor_interleaved \
    --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
    --temporal-dt-rank auto --temporal-layers 2 \
    --n-samples 32 \
    --out-json "${OUT}"
done
