#!/usr/bin/env bash
# Merge and rank the completed eight-anchor cosine-pilot screens.
#SBATCH --job-name=v22f-r1-cos-summary
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/v22f_r1_cos_summary_%j.out
#SBATCH --error=logs/v22f_r1_cos_summary_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-v22f-r1-cos-summary-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
source scripts/graphcast_env.sh

python -u scripts/analyze_models/summarize_v22_final_res1_cosine_screen.py
