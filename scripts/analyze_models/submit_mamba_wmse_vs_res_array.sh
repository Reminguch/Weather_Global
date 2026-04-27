#!/bin/bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

RESOLUTIONS="${RESOLUTIONS:-$(find artifacts/checkpoints/graphcast_stream_frozen_residual_mamba -maxdepth 1 -mindepth 1 -type d | sed 's#.*/##' | sed -n 's/.*_res\([0-9][0-9]*\)_.*/\1/p' | sort -n | uniq | xargs)}"
WARMUP_STEPS="${WARMUP_STEPS:-24}"
TRUNK_STEPS="${TRUNK_STEPS:-32}"
if [[ -z "${RESOLUTIONS}" ]]; then
  echo "No mamba resolutions discovered." >&2
  exit 1
fi

read -r -a RES_ARRAY <<< "${RESOLUTIONS}"
ARRAY_MAX=$((${#RES_ARRAY[@]} - 1))

ARRAY_JOB_ID=$(
  sbatch \
    --parsable \
    --array="0-${ARRAY_MAX}" \
    --export=ALL,RESOLUTIONS="${RESOLUTIONS}",WARMUP_STEPS="${WARMUP_STEPS}",TRUNK_STEPS="${TRUNK_STEPS}" \
    scripts/analyze_models/run_mamba_wmse_vs_res_array.slurm
)
echo "Submitted array job: ${ARRAY_JOB_ID} for resolutions: ${RESOLUTIONS}"

MERGE_JOB_ID=$(
  sbatch \
    --parsable \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --export=ALL,RESOLUTIONS="${RESOLUTIONS}",WARMUP_STEPS="${WARMUP_STEPS}",TRUNK_STEPS="${TRUNK_STEPS}" \
    scripts/analyze_models/run_mamba_wmse_vs_res_merge.slurm
)
echo "Submitted merge job: ${MERGE_JOB_ID}"
