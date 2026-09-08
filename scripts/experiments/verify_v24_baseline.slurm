#!/usr/bin/env bash
#SBATCH --job-name=v24-cache-verify
#SBATCH --account=dabanin
#SBATCH --qos=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v24_baseline_verify_%j.out
#SBATCH --error=logs/v24_baseline_verify_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/graphcast_env.sh
export JAX_PLATFORMS=cpu
python -u scripts/preprocessing/precompute_v24_baseline.py \
  --config "${1:?config}" --output-root "${2:?cache root}" --num-shards 8 --verify
