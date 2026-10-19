#!/usr/bin/env bash
#SBATCH --job-name=v24-ar-summary
#SBATCH --account=dabanin
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/v24_ar_summary_%j.out
#SBATCH --error=logs/v24_ar_summary_%j.err
set -euo pipefail
EXPERIMENT="${1:?experiment required}"
WORKSPACE="${2:?workspace required}"
source "${EXPERIMENT}/source/scripts/experiments/run_frozen_experiment.sh"
run_frozen_experiment "${WORKSPACE}" "${EXPERIMENT}" \
  scripts/experiments/run_v24_res1_ar_sweep.py summarize 0
