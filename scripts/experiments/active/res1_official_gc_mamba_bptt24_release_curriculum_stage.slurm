#!/bin/bash
#SBATCH --job-name=res1-gcm-release-stage
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --output=logs/res1_gc_mamba_release_curriculum_%x_%j.out
#SBATCH --error=logs/res1_gc_mamba_release_curriculum_%x_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
mkdir -p .tmp/matplotlib
export MPLCONFIGDIR="${PROJECT_ROOT}/.tmp/matplotlib"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

source scripts/graphcast_env.sh

DATA_PATH=${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}
DATA_SOURCE=${DATA_SOURCE:-prepared_array}
PREPARED_DATA_ROOT=${PREPARED_DATA_ROOT:-data/graphcast/graphcast/dataset/prepared_stream}
STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}
OFFICIAL_CKPT=${OFFICIAL_CKPT:-"data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"}

OUT_DIR=${OUT_DIR:-artifacts/checkpoints/res1_official/gc_mamba_bptt24_release_curriculum_paperlr}
STAGE_NAME=${STAGE_NAME:-stage}
TARGET_STEPS=${TARGET_STEPS:-4}
TRAINABLE_PART=${TRAINABLE_PART:-mamba}
MAX_STEPS=${MAX_STEPS:-10000}
RESUME_STEP=${RESUME_STEP:-}

RESOLUTION=${RESOLUTION:-1.0}
MESH_SIZE=${MESH_SIZE:-5}
WIDTH=${WIDTH:-512}
MSG=${MSG:-16}
INPUT_DURATION=${INPUT_DURATION:-12h}
VAL_YEAR=${VAL_YEAR:-2022}
TRAIN_START_YEAR=${TRAIN_START_YEAR:-2015}
TRAIN_END_YEAR=${TRAIN_END_YEAR:-2021}

LEN_SEGMENT=${LEN_SEGMENT:-96}
BPTT_STEPS=${BPTT_STEPS:-24}
CHUNK_LOAD_WORKERS=${CHUNK_LOAD_WORKERS:-6}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
EVAL_NUM_SEGMENTS=${EVAL_NUM_SEGMENTS:-16}
FINAL_EVAL_NUM_SEGMENTS=${FINAL_EVAL_NUM_SEGMENTS:-all}
EVAL_EVERY=${EVAL_EVERY:-2000}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-2000}
DATA_CACHE_MODE=${DATA_CACHE_MODE:-never}
DATA_CACHE_MAX_GIB=${DATA_CACHE_MAX_GIB:-48}
BATCH_BUILDER=${BATCH_BUILDER:-prepared_array}

LR=${LR:-1e-4}
GRAPHCAST_LR=${GRAPHCAST_LR:-}
MAMBA_LR=${MAMBA_LR:-}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
ADAMW_BETA1=${ADAMW_BETA1:-0.9}
ADAMW_BETA2=${ADAMW_BETA2:-0.999}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-}

PRECISION=${PRECISION:-bf16}
SEED=${SEED:-0}
TEMPORAL_INSERT_COUNT=${TEMPORAL_INSERT_COUNT:-1}
TEMPORAL_LAYERS=${TEMPORAL_LAYERS:-2}
D_INNER=${D_INNER:-128}
D_STATE=${D_STATE:-16}
MEMORY_MODE=${MEMORY_MODE:-optimal}
AR_LOSS_MODE=${AR_LOSS_MODE:-all_bptt_uniform}
DRY_RUN=${DRY_RUN:-0}

CKPT_IN=${CKPT_IN:-${OFFICIAL_CKPT}}
INIT_FROM_GRAPHCAST_CKPT=${INIT_FROM_GRAPHCAST_CKPT:-}
if [[ "${INIT_FROM_GRAPHCAST_CKPT}" == "official" ]]; then
  INIT_FROM_GRAPHCAST_CKPT="${OFFICIAL_CKPT}"
fi

RUN_NAME=${RUN_NAME:-"official_res1_gcsmall_gc_mamba_${STAGE_NAME}_k${TARGET_STEPS}_bptt${BPTT_STEPS}_ls${LEN_SEGMENT}_${TRAINABLE_PART}"}
RUN_DIR="${OUT_DIR}/${RUN_NAME}"

if [[ ! -f "${CKPT_IN}" ]]; then
  echo "Missing input checkpoint: ${CKPT_IN}" >&2
  exit 1
fi
if [[ -n "${INIT_FROM_GRAPHCAST_CKPT}" && ! -f "${INIT_FROM_GRAPHCAST_CKPT}" ]]; then
  echo "Missing init-from checkpoint: ${INIT_FROM_GRAPHCAST_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${PREPARED_DATA_ROOT}/res1" ]]; then
  echo "Missing prepared res1 data store: ${PREPARED_DATA_ROOT}/res1" >&2
  exit 1
fi

echo "Stage ${STAGE_NAME}: target_steps=${TARGET_STEPS} trainable_part=${TRAINABLE_PART} max_steps=${MAX_STEPS}"
echo "Run directory: ${RUN_DIR}"
echo "Checkpoint in: ${CKPT_IN}"
echo "LRs: lr=${LR} graphcast_lr=${GRAPHCAST_LR:-<unset>} mamba_lr=${MAMBA_LR:-<unset>}"
echo "AdamW: weight_decay=${WEIGHT_DECAY} beta1=${ADAMW_BETA1} beta2=${ADAMW_BETA2} max_grad_norm=${MAX_GRAD_NORM:-<unset>}"

CMD=(
  python -u -m src.models.mamba.training.segments_training
  --model gc_mamba
  --data-path "${DATA_PATH}"
  --data-source "${DATA_SOURCE}"
  --prepared-data-root "${PREPARED_DATA_ROOT}"
  --stats-dir "${STATS_DIR}"
  --out-dir "${OUT_DIR}"
  --run-name "${RUN_NAME}"
  --val-year "${VAL_YEAR}"
  --train-start-year "${TRAIN_START_YEAR}"
  --train-end-year "${TRAIN_END_YEAR}"
  --resolution "${RESOLUTION}"
  --mesh-size "${MESH_SIZE}"
  --width "${WIDTH}"
  --processor-msg-steps "${MSG}"
  --input-duration "${INPUT_DURATION}"
  --target-steps "${TARGET_STEPS}"
  --len-segment "${LEN_SEGMENT}"
  --bptt-steps "${BPTT_STEPS}"
  --autoregressive-loss-mode "${AR_LOSS_MODE}"
  --chunk-load-workers "${CHUNK_LOAD_WORKERS}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --eval-num-segments "${EVAL_NUM_SEGMENTS}"
  --final-eval-num-segments "${FINAL_EVAL_NUM_SEGMENTS}"
  --data-cache-mode "${DATA_CACHE_MODE}"
  --data-cache-max-gib "${DATA_CACHE_MAX_GIB}"
  --batch-builder "${BATCH_BUILDER}"
  --max-steps "${MAX_STEPS}"
  --eval-every "${EVAL_EVERY}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --adamw-beta1 "${ADAMW_BETA1}"
  --adamw-beta2 "${ADAMW_BETA2}"
  --precision "${PRECISION}"
  --seed "${SEED}"
  --memory-mode "${MEMORY_MODE}"
  --temporal-backbone mamba
  --temporal-location mesh_processor_interleaved
  --temporal-stateful
  --temporal-d-inner "${D_INNER}"
  --temporal-d-state "${D_STATE}"
  --temporal-d-conv 4
  --temporal-dt-rank auto
  --temporal-layers "${TEMPORAL_LAYERS}"
  --temporal-dropout 0.0
  --temporal-insert-count "${TEMPORAL_INSERT_COUNT}"
  --ckpt-in "${CKPT_IN}"
  --trainable-part "${TRAINABLE_PART}"
  --zero-init-temporal-out
)

if [[ -n "${GRAPHCAST_LR}" ]]; then
  CMD+=(--graphcast-lr "${GRAPHCAST_LR}")
fi
if [[ -n "${MAMBA_LR}" ]]; then
  CMD+=(--mamba-lr "${MAMBA_LR}")
fi
if [[ -n "${MAX_GRAD_NORM}" ]]; then
  CMD+=(--max-grad-norm "${MAX_GRAD_NORM}")
fi
if [[ -n "${RESUME_STEP}" ]]; then
  CMD+=(--resume-step "${RESUME_STEP}")
fi
if [[ -n "${INIT_FROM_GRAPHCAST_CKPT}" ]]; then
  CMD+=(--init-from-graphcast-ckpt "${INIT_FROM_GRAPHCAST_CKPT}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Resolved command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
