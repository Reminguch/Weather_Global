#!/usr/bin/env bash
# Sanity check: run v17 two-stream eval on v15 v2 step 20k ckpt.
# Goals:
#  1) K=1 lead RMSE must match v15 v2's old K=12 rollout K=1 entry (bf16-roundoff).
#     Proves v17 doesn't break the non-AR path.
#  2) two_stream_rollout debug prints will show stream divergence:
#     k=0 zero, k>=1 nonzero. Proves the two streams really diverge.
#  3) K=2..12 values may differ from v15 v2 (v15 v2 was a single-stream eval).
#     That's the whole point of v17.

#SBATCH --job-name=v17-sanity
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v17_sanity_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v17_sanity_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v17/eval_v17_rollout.py
cd /home/lm8598/Weather_Global_experiments

CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v15_v13arch_7yr_v2/v15_20k_v2
EVAL_OUT_DIR=${CKPT_BASE}/eval_results_v17
mkdir -p "${EVAL_OUT_DIR}"

# Run v17 two-stream eval on the v15 v2 K=1-trained ckpt.
${PYTHON} -u ${SCRIPT} \
  --ckpt "${CKPT_BASE}/v13_residual_step20000.pkl" \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 12 \
  --n-samples 32 \
  --out-json "${EVAL_OUT_DIR}/v17_check_v15v2_step20000_K12.json"
