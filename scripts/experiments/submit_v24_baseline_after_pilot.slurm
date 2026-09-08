#!/usr/bin/env bash
#SBATCH --job-name=v24-cache-release
#SBATCH --account=dabanin
#SBATCH --qos=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/v24_baseline_release_%j.out
#SBATCH --error=logs/v24_baseline_release_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/graphcast_env.sh
export JAX_PLATFORMS=cpu
python -u scripts/preprocessing/submit_v24_baseline_production.py \
  --config "${1:?config}" --output-root "${2:?production root}" \
  --pilot-root "${3:?pilot root}" --pilot-job-id "${4:?pilot job}"
