#!/bin/bash
#SBATCH --job-name=gcm-r4-smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/gcm_r4_smoke_%j.out
#SBATCH --error=logs/gcm_r4_smoke_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
source scripts/graphcast_env.sh

MODEL=gc_mamba

DATA_PATH=${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}
STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}
CKPT_IN=${CKPT_IN:-"data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"}
OUT_DIR=${OUT_DIR:-artifacts/checkpoints/experiments_smoke_gc_mamba_res4}

VAL_YEAR=${VAL_YEAR:-2021}
TRAIN_START_YEAR=${TRAIN_START_YEAR:-}
TRAIN_END_YEAR=${TRAIN_END_YEAR:-}

RESOLUTION=4
MESH_SIZE=4
WIDTH=128
MSG=1
INPUT_DURATION=${INPUT_DURATION:-12h}
TARGET_STEPS=${TARGET_STEPS:-1}
LEN_SEGMENT=32
BPTT_STEPS=${BPTT_STEPS:-8}
CHUNK_LOAD_WORKERS=${CHUNK_LOAD_WORKERS:-6}

BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
MAX_STEPS=2000
EVAL_EVERY=500
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-500}
DATA_CACHE_MODE=${DATA_CACHE_MODE:-auto}
DATA_CACHE_MAX_GIB=${DATA_CACHE_MAX_GIB:-48}
BATCH_BUILDER=${BATCH_BUILDER:-numpy}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
PRECISION=${PRECISION:-bf16}
SEED=${SEED:-0}

TEMPORAL_BACKBONE=mamba
TEMPORAL_LOCATION=${TEMPORAL_LOCATION:-mesh_processor_interleaved}
TEMPORAL_D_INNER=${TEMPORAL_D_INNER:-16}
TEMPORAL_D_STATE=${TEMPORAL_D_STATE:-16}
TEMPORAL_D_CONV=${TEMPORAL_D_CONV:-4}
TEMPORAL_DT_RANK=${TEMPORAL_DT_RANK:-auto}
TEMPORAL_LAYERS=1
TEMPORAL_DROPOUT=${TEMPORAL_DROPOUT:-0.0}
TEMPORAL_STATEFUL=${TEMPORAL_STATEFUL:-1}

RUN_NAME=${RUN_NAME:-gc_mamba_res4_m4_w128_mp1_dh16_ls32_smoke}

CMD=(
  python -u -m src.models.mamba.training.segments_training
  --model "${MODEL}"
  --data-path "${DATA_PATH}"
  --stats-dir "${STATS_DIR}"
  --out-dir "${OUT_DIR}"
  --run-name "${RUN_NAME}"
  --val-year "${VAL_YEAR}"
  --resolution "${RESOLUTION}"
  --mesh-size "${MESH_SIZE}"
  --width "${WIDTH}"
  --processor-msg-steps "${MSG}"
  --input-duration "${INPUT_DURATION}"
  --target-steps "${TARGET_STEPS}"
  --len-segment "${LEN_SEGMENT}"
  --bptt-steps "${BPTT_STEPS}"
  --chunk-load-workers "${CHUNK_LOAD_WORKERS}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --data-cache-mode "${DATA_CACHE_MODE}"
  --data-cache-max-gib "${DATA_CACHE_MAX_GIB}"
  --batch-builder "${BATCH_BUILDER}"
  --max-steps "${MAX_STEPS}"
  --eval-every "${EVAL_EVERY}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --precision "${PRECISION}"
  --seed "${SEED}"
  --temporal-backbone "${TEMPORAL_BACKBONE}"
  --temporal-location "${TEMPORAL_LOCATION}"
  --temporal-d-inner "${TEMPORAL_D_INNER}"
  --temporal-d-state "${TEMPORAL_D_STATE}"
  --temporal-d-conv "${TEMPORAL_D_CONV}"
  --temporal-dt-rank "${TEMPORAL_DT_RANK}"
  --temporal-layers "${TEMPORAL_LAYERS}"
  --temporal-dropout "${TEMPORAL_DROPOUT}"
  --ckpt-in "${CKPT_IN}"
)

if [[ "${TEMPORAL_STATEFUL}" == "1" ]]; then
  CMD+=(--temporal-stateful)
fi

if [[ -n "${TRAIN_START_YEAR}" && -n "${TRAIN_END_YEAR}" ]]; then
  CMD+=(--train-start-year "${TRAIN_START_YEAR}" --train-end-year "${TRAIN_END_YEAR}")
fi

"${CMD[@]}"
