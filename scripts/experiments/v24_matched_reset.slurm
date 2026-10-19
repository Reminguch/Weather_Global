#!/usr/bin/env bash
# Principal 13740084_6: 22h41m; sibling peak 234.8 GiB. 30h/320G retain headroom.
#SBATCH --job-name=v24-selected-reset
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=1-06:00:00
#SBATCH --output=logs/v24_selected_reset_%j.out
#SBATCH --error=logs/v24_selected_reset_%j.err
set -euo pipefail
WORKSPACE="${2:-${SLURM_SUBMIT_DIR:?workspace required}}"
EXPERIMENT="${1:?prepared experiment required}"
source "${EXPERIMENT}/source/scripts/experiments/run_frozen_experiment.sh"
run_frozen_experiment "${WORKSPACE}" "${EXPERIMENT}" \
  scripts/experiments/run_v24_matched_reset.py train 0
