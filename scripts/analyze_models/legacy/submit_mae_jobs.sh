#!/usr/bin/env bash
set -euo pipefail

# Submit one job per model group.
# Optional overrides:
#   N_EVAL_DAYS=40 WINDOW_BATCH_SIZE=8 DATASET_DIR=... ./scripts/analyze_models/legacy/submit_mae_jobs.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

for GROUP in res2 res4 baseline; do
  echo "Submitting ${GROUP}..."
  sbatch --export=ALL,MODEL_GROUP="${GROUP}" scripts/analyze_models/legacy/run_mae_vs_lead.slurm
done
