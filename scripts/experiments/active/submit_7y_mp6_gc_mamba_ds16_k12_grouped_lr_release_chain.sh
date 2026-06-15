#!/bin/bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs

DRY_RUN=${DRY_RUN:-0}
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

TRAIN_SCRIPT="scripts/experiments/active/7y_mp6_gc_mamba_ds16_k12_grouped_lr_release20k.slurm"
EVAL_SCRIPT="scripts/analyze_models/submit_res2_ds16_gc_mamba_k12_grouped_lr_release_eval.sh"

TRAIN_CMD=(
  sbatch
  --parsable
  --job-name=7y-mp6-r2-ds16-k12-glr
  --mem=96G
  --time=07:00:00
  --array=0-1
  "${TRAIN_SCRIPT}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY RUN training:'
  printf ' %q' "${TRAIN_CMD[@]}"
  printf '\n'
  echo "DRY RUN eval wrapper: UPSTREAM_DEPENDENCY=<training_job_id> DRY_RUN=1 bash ${EVAL_SCRIPT}"
  exit 0
fi

training_job_raw=$("${TRAIN_CMD[@]}")
training_job_id="${training_job_raw%%;*}"
if [[ ! "${training_job_id}" =~ ^[0-9]+$ ]]; then
  echo "Training sbatch did not return a valid job id: ${training_job_raw}" >&2
  exit 1
fi
echo "Submitted training array: ${training_job_id}"

UPSTREAM_DEPENDENCY="${training_job_id}" bash "${EVAL_SCRIPT}"
