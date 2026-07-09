#!/usr/bin/env bash
# Mamba functional ablation on the step-2000 GC-Mamba_v2 ckpt.
# 4 modes: normal / reset_state / zero_D / reset_state+zero_D.
# 32 val anchors, K=1 truth-fed, bptt=24.
# Verdict guide:
#   reset_state ≈ normal     → Mamba state NOT used (memory not helping)
#   zero_D     >> normal     → D path was carrying the model
#   reset_state >> normal AND zero_D ≈ normal → recurrent state IS doing work
#
# WARNING: 2000 step is early — Mamba may not have learned much yet.
# We'll re-run on step 5000/10000/20000 ckpts when 20k training finishes.
#SBATCH --job-name=gcmv2-ablation-s2000
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/gcmv2_ablation_s2000_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/gcmv2_ablation_s2000_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

CKPT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/gcmamba_v2/gcmamba_v2_postproc_K1_2k/ckpt_step2000.npz
OUT_JSON=/home/lm8598/Weather_Global_experiments/results/gcmamba_v2_diagnostics/ablation_step2000.json
mkdir -p "$(dirname ${OUT_JSON})"

$PYTHON -u scripts/training/gcmamba_v2_diagnostics/ablation_eval.py \
  --ckpt "${CKPT}" \
  --temporal-location mesh_post_processor \
  --temporal-insert-count 1 \
  --temporal-d-inner 128 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 1 \
  --bptt-steps 24 --n-anchors 32 \
  --ablations "normal,reset_state,zero_D,reset_state+zero_D" \
  --out-json "${OUT_JSON}"

echo "ablation eval at step 2000 done"
