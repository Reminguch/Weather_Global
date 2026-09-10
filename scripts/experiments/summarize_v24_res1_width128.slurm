#!/usr/bin/env bash
#SBATCH --job-name=v24-width-summary
#SBATCH --account=dabanin
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/v24_width_summary_%j.out
#SBATCH --error=logs/v24_width_summary_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
EXPERIMENT="${1:?absolute experiment directory required}"
export PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
source "${EXPERIMENT}/source/scripts/graphcast_env.sh"
export PYTHONPATH="${EXPERIMENT}/source:${EXPERIMENT}/source/third_party/graphcast"
export MPLCONFIGDIR="/tmp/mpl-v24-width-${SLURM_JOB_ID}"
python -u "${EXPERIMENT}/source/scripts/experiments/run_v24_res1_width128.py" \
  summarize --experiment "${EXPERIMENT}"
