#!/bin/bash
#SBATCH --job-name=seg8y-mp8
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/seg8y_mp8_%A_%a.out
#SBATCH --error=logs/seg8y_mp8_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
RESOLUTIONS=(2.0 4.0 9.0 15.0)
MESH_SIZES=(4 3 2 2)
RESOLUTION="${RESOLUTIONS[$TASK_ID]}"
MESH_SIZE="${MESH_SIZES[$TASK_ID]}"

export BATCH_SIZE=${BATCH_SIZE:-8}
SUFFIX=""
if [[ "${BATCH_SIZE}" == "4" ]]; then
  SUFFIX="_bs4"
fi
export RUN_NAME=${RUN_NAME:-base_seg_8y_res${RESOLUTION%.*}_m${MESH_SIZE}_w512_mp8_ls30_bptt6_t1${SUFFIX}}

SLURM_ARRAY_TASK_ID=$((TASK_ID * 3)) \
  MPS_OVERRIDE=8 \
  bash scripts/experiments/archive/2026-05_8y_w512_pilots/graphcast_segments_8y_w512_mp124.slurm
