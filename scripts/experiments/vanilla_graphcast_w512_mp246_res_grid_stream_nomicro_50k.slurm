#!/bin/bash
#SBATCH --job-name=gc-w512-nomb
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/gc_w512_nomicro_stream_50k_%A_%a.out
#SBATCH --error=logs/gc_w512_nomicro_stream_50k_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
source scripts/graphcast_env.sh

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

RESOLUTIONS=(2.0 4.0 9.0 15.0)
MESH_SIZES=(4 3 2 2)
MPS=(2 4 6)
TRAIN_WINDOWS=("2019:2021:3y" "2015:2021:7y")

COMBOS_PER_WINDOW=$((${#RESOLUTIONS[@]} * ${#MPS[@]}))
WINDOW_IDX=$((TASK_ID / COMBOS_PER_WINDOW))
REM=$((TASK_ID % COMBOS_PER_WINDOW))
RES_IDX=$((REM / ${#MPS[@]}))
MP_IDX=$((REM % ${#MPS[@]}))

IFS=: read -r TRAIN_START_YEAR TRAIN_END_YEAR TRAIN_TAG <<< "${TRAIN_WINDOWS[$WINDOW_IDX]}"
RESOLUTION="${RESOLUTIONS[$RES_IDX]}"
MESH_SIZE="${MESH_SIZES[$RES_IDX]}"
MSG="${MPS[$MP_IDX]}"

DATA_PATH=${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}
DATA_SOURCE=${DATA_SOURCE:-prepared_array}
PREPARED_DATA_ROOT=${PREPARED_DATA_ROOT:-data/graphcast/graphcast/dataset/prepared_stream}
STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}
CKPT_IN=${CKPT_IN:-"data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"}
OUT_DIR=${OUT_DIR:-artifacts/checkpoints/vanilla_graphcast_w512_mp246_stream_nomicro_50k}

WIDTH=512
INPUT_DURATION=${INPUT_DURATION:-12h}
TARGET_STEPS=${TARGET_STEPS:-1}
VAL_YEAR=${VAL_YEAR:-2022}

BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
MAX_STEPS=${MAX_STEPS:-50000}
EVAL_EVERY=${EVAL_EVERY:-2000}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-2000}
DATA_CACHE_MODE=${DATA_CACHE_MODE:-never}
DATA_CACHE_MAX_GIB=${DATA_CACHE_MAX_GIB:-48}
BATCH_BUILDER=${BATCH_BUILDER:-prepared_array}
PREFETCH_WORKERS=${PREFETCH_WORKERS:-4}
PREFETCH_DEPTH=${PREFETCH_DEPTH:-8}
PREFETCH_DEVICE_DEPTH=${PREFETCH_DEVICE_DEPTH:-1}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
PRECISION=${PRECISION:-bf16}
SEED=${SEED:-0}

RUN_NAME=${RUN_NAME:-vanilla_gc_${TRAIN_TAG}_res${RESOLUTION%.*}_m${MESH_SIZE}_w${WIDTH}_mp${MSG}_h6_bs${BATCH_SIZE}_accum${GRAD_ACCUM_STEPS}_stream50k}

echo "Submitting task ${TASK_ID}: train=${TRAIN_START_YEAR}-${TRAIN_END_YEAR}, val=${VAL_YEAR}, res=${RESOLUTION}, mesh=${MESH_SIZE}, mp=${MSG}, run=${RUN_NAME}"

python -u -m src.models.graphcast.training.standard_training \
  --data-path "${DATA_PATH}" \
  --data-source "${DATA_SOURCE}" \
  --prepared-data-root "${PREPARED_DATA_ROOT}" \
  --stats-dir "${STATS_DIR}" \
  --ckpt-in "${CKPT_IN}" \
  --out-dir "${OUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --val-year "${VAL_YEAR}" \
  --train-start-year "${TRAIN_START_YEAR}" \
  --train-end-year "${TRAIN_END_YEAR}" \
  --resolution "${RESOLUTION}" \
  --mesh-size "${MESH_SIZE}" \
  --width "${WIDTH}" \
  --processor-msg-steps "${MSG}" \
  --input-duration "${INPUT_DURATION}" \
  --target-steps "${TARGET_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --data-cache-mode "${DATA_CACHE_MODE}" \
  --data-cache-max-gib "${DATA_CACHE_MAX_GIB}" \
  --batch-builder "${BATCH_BUILDER}" \
  --prefetch-workers "${PREFETCH_WORKERS}" \
  --prefetch-depth "${PREFETCH_DEPTH}" \
  --prefetch-device-depth "${PREFETCH_DEVICE_DEPTH}" \
  --max-steps "${MAX_STEPS}" \
  --eval-every "${EVAL_EVERY}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --precision "${PRECISION}" \
  --seed "${SEED}"
