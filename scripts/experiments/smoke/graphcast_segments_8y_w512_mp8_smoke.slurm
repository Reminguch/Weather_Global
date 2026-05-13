#!/bin/bash
#SBATCH --job-name=seg8y-mp8-smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/seg8y_mp8_smoke_%A_%a.out
#SBATCH --error=logs/seg8y_mp8_smoke_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
source scripts/graphcast_env.sh

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
RESOLUTIONS=(2.0 4.0 9.0 15.0)
MESH_SIZES=(4 3 2 2)
RESOLUTION="${RESOLUTIONS[$TASK_ID]}"
MESH_SIZE="${MESH_SIZES[$TASK_ID]}"

export OUT_DIR=${OUT_DIR:-artifacts/checkpoints/8_years/smoke_graphcast_segments_w512_mp8}
export MAX_STEPS=${MAX_STEPS:-20}
export EVAL_EVERY=${EVAL_EVERY:-1000000}
export CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-1000000}
export BATCH_SIZE=${BATCH_SIZE:-8}
export RUN_NAME=${RUN_NAME:-base_seg_8y_res${RESOLUTION%.*}_m${MESH_SIZE}_w512_mp8_ls30_bptt6_t1_smoke}

SLURM_ARRAY_TASK_ID=$((TASK_ID * 3)) \
  MPS_OVERRIDE=8 \
  bash scripts/experiments/archive/2026-05_8y_w512_pilots/graphcast_segments_8y_w512_mp124.slurm
