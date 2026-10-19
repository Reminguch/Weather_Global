#!/usr/bin/env bash
# 13740084_6-9: 20.2-22.7h; comparable batch4 runs peak near 235GiB RSS.
# Fixed BPTT24 at every K; allow extra truth loading for the shorter AR tails.
#SBATCH --job-name=v24-ar-train
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/v24_ar_%A_%a.out
#SBATCH --error=logs/v24_ar_%A_%a.err
set -euo pipefail
WORKSPACE="${3:?workspace required}"
EXPERIMENT="${2:?experiment required}"
source "${EXPERIMENT}/source/scripts/experiments/run_frozen_experiment.sh"
run_frozen_experiment "${WORKSPACE}" "${EXPERIMENT}" \
  scripts/experiments/run_v24_res1_ar_sweep.py "${1:?stage required}" "${SLURM_ARRAY_TASK_ID:-0}"
