#!/usr/bin/env bash
# Render the matched d_conv=4 stateful/stateless rollout comparison after all
# four JSON evaluation outputs have been produced.
#SBATCH --job-name=resm-r2-dc4-plot
#SBATCH --account=dabanin
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/resm_r2_dc4_plot_%j.out
#SBATCH --error=logs/resm_r2_dc4_plot_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT}"

mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-weather-global-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
source scripts/graphcast_env.sh

python -u scripts/analyze_models/plot_resmamba_dc4_stateful_stateless.py
