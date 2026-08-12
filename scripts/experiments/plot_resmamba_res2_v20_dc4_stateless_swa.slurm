#!/usr/bin/env bash
# Render the matched open-loop stateful/stateless SWA(4k--12k) comparison.
#SBATCH --job-name=resm-r2-dc4-stat-swa-plot
#SBATCH --account=dabanin
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/resm_r2_dc4_stateless_swa_plot_%j.out
#SBATCH --error=logs/resm_r2_dc4_stateless_swa_plot_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-weather-global-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"

source scripts/graphcast_env.sh
python -u scripts/analyze_models/plot_resmamba_dc4_stateful_stateless.py --swa-4k-12k
