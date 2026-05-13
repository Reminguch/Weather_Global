#!/bin/bash
#SBATCH --job-name=gc7y-mamba
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/7y_small_mamba_%x_%A_%a.out
#SBATCH --error=logs/7y_small_mamba_%x_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
source scripts/graphcast_env.sh

RES_KEY=${RES_KEY:?Set RES_KEY to one of: res2, res3, res6}
STAGE=${STAGE:-frozen}
case "${RES_KEY}" in
  res2)
    BASE_NAME="vanilla_gc_7y_res2_m4_w256_mp2_h6_bs8_accum1_stream200k"
    ;;
  res3)
    BASE_NAME="vanilla_gc_7y_res3_m4_w256_mp2_h6_bs8_accum1_stream200k"
    ;;
  res6)
    BASE_NAME="vanilla_gc_7y_res6_m3_w256_mp2_h6_bs8_accum1_stream200k"
    ;;
  *)
    echo "Unknown RES_KEY=${RES_KEY}; expected res2, res3, or res6." >&2
    exit 1
    ;;
esac

case "${STAGE}" in
  frozen|release) ;;
  *)
    echo "Unknown STAGE=${STAGE}; expected frozen or release." >&2
    exit 1
    ;;
esac

D_INNERS=(128 256)
D_STATES=(64 128)
VARIANTS=(
  "gc_mamba:2"
  "residual_mamba:2"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
D_COMBOS=$((${#D_INNERS[@]} * ${#D_STATES[@]}))
EXPECTED_TASKS=$((${#VARIANTS[@]} * D_COMBOS))
if (( TASK_ID < 0 || TASK_ID >= EXPECTED_TASKS )); then
  echo "Task ${TASK_ID} outside expected range 0-$((EXPECTED_TASKS - 1))" >&2
  exit 1
fi

VARIANT_IDX=$((TASK_ID / D_COMBOS))
D_IDX=$((TASK_ID % D_COMBOS))
D_INNER_IDX=$((D_IDX / ${#D_STATES[@]}))
D_STATE_IDX=$((D_IDX % ${#D_STATES[@]}))

IFS=: read -r MODEL INSERT_COUNT <<< "${VARIANTS[$VARIANT_IDX]}"
D_INNER="${D_INNERS[$D_INNER_IDX]}"
D_STATE="${D_STATES[$D_STATE_IDX]}"

BASE_ROOT=${BASE_ROOT:-artifacts/checkpoints/7_years/small_experiments}
BASE_DIR="${BASE_ROOT}/${BASE_NAME}"
BASE_CKPT="${BASE_DIR}/ckpt_step200000.npz"
RUN_CONFIG="${BASE_DIR}/run_config.json"

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Missing exact 200k baseline checkpoint: ${BASE_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${RUN_CONFIG}" ]]; then
  echo "Missing baseline run config: ${RUN_CONFIG}" >&2
  exit 1
fi

readarray -t CFG < <(python - "${RUN_CONFIG}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)

model = cfg["model_config"]
task = cfg["task_config"]
pipeline = cfg["data_pipeline"]
fields = [
    cfg["data_path"],
    str(cfg.get("data_source", pipeline.get("data_source", "prepared_array"))),
    str(cfg.get("prepared_data_root", pipeline.get("prepared_data_root", ""))),
    str(cfg["val_year"]),
    "" if cfg.get("train_start_year") is None else str(cfg["train_start_year"]),
    "" if cfg.get("train_end_year") is None else str(cfg["train_end_year"]),
    str(model["resolution"]),
    str(model["mesh_size"]),
    str(model["latent_size"]),
    str(model["gnn_msg_steps"]),
    str(task["input_duration"]),
    str(cfg["batch_size"]),
    str(cfg["eval_batch_size"]),
    str(cfg.get("eval_every", 2000)),
    str(cfg.get("checkpoint_every", 2000)),
    str(pipeline.get("data_cache_mode", "never")),
    str(pipeline.get("data_cache_max_gib", 48)),
    str(pipeline.get("batch_builder", "prepared_array")),
    str(cfg.get("lr", 1e-4)),
    str(cfg.get("weight_decay", 1e-4)),
    str(cfg.get("precision", "bf16")),
    str(cfg.get("seed", 0)),
]
print("\n".join(fields))
PY
)

DATA_PATH="${CFG[0]}"
DATA_SOURCE="${CFG[1]}"
PREPARED_DATA_ROOT="${CFG[2]}"
VAL_YEAR="${CFG[3]}"
TRAIN_START_YEAR="${CFG[4]}"
TRAIN_END_YEAR="${CFG[5]}"
RESOLUTION="${CFG[6]}"
MESH_SIZE="${CFG[7]}"
WIDTH="${CFG[8]}"
MSG="${CFG[9]}"
INPUT_DURATION="${CFG[10]}"
BATCH_SIZE="${CFG[11]}"
EVAL_BATCH_SIZE="${CFG[12]}"
EVAL_EVERY="${CFG[13]}"
CHECKPOINT_EVERY="${CFG[14]}"
DATA_CACHE_MODE="${CFG[15]}"
DATA_CACHE_MAX_GIB="${CFG[16]}"
BATCH_BUILDER="${CFG[17]}"
LR="${CFG[18]}"
WEIGHT_DECAY="${CFG[19]}"
PRECISION="${CFG[20]}"
SEED="${CFG[21]}"

STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}
FROZEN_OUT_DIR=${FROZEN_OUT_DIR:-artifacts/checkpoints/7_years/small_mamba_frozen_from_vanilla_200k_50k}
RELEASE_OUT_DIR=${RELEASE_OUT_DIR:-artifacts/checkpoints/7_years/small_mamba_release_from_frozen50k_20k}
LEN_SEGMENT=${LEN_SEGMENT:-30}
BPTT_STEPS=${BPTT_STEPS:-6}
CHUNK_LOAD_WORKERS=${CHUNK_LOAD_WORKERS:-6}
EVAL_NUM_SEGMENTS=${EVAL_NUM_SEGMENTS:-16}
FINAL_EVAL_NUM_SEGMENTS=${FINAL_EVAL_NUM_SEGMENTS:-all}
BATCH_SIZE=${BATCH_SIZE_OVERRIDE:-${BATCH_SIZE}}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE_OVERRIDE:-${EVAL_BATCH_SIZE}}
RESUME_FROM_LATEST=${RESUME_FROM_LATEST:-1}

FROZEN_NAME="${BASE_NAME}_${MODEL}_tc${INSERT_COUNT}_di${D_INNER}_ds${D_STATE}_frozen50k"

if [[ "${STAGE}" == "frozen" ]]; then
  OUT_DIR="${FROZEN_OUT_DIR}"
  RUN_NAME="${FROZEN_NAME}"
  RUN_DIR="${OUT_DIR}/${RUN_NAME}"
  MAX_STEPS=${MAX_STEPS:-50000}
  TRAINABLE_PART="mamba"
  START_CKPT="${BASE_CKPT}"
  START_STEP=""
else
  OUT_DIR="${RELEASE_OUT_DIR}"
  RUN_NAME="${FROZEN_NAME}_release20k"
  RUN_DIR="${OUT_DIR}/${RUN_NAME}"
  MAX_STEPS=${MAX_STEPS:-70000}
  TRAINABLE_PART="all"
  FROZEN_CKPT="${FROZEN_OUT_DIR}/${FROZEN_NAME}/ckpt_step50000.npz"
  if [[ ! -f "${FROZEN_CKPT}" ]]; then
    echo "Missing exact frozen 50k checkpoint: ${FROZEN_CKPT}" >&2
    exit 1
  fi
  START_CKPT="${FROZEN_CKPT}"
  START_STEP="50000"
fi

echo "Task ${TASK_ID}/${EXPECTED_TASKS}: stage=${STAGE} base=${BASE_NAME} model=${MODEL} insert=${INSERT_COUNT} d_inner=${D_INNER} d_state=${D_STATE}"

RESUME_STEP=""
RESUME_CKPT=""
if [[ "${RESUME_FROM_LATEST}" == "1" && -d "${RUN_DIR}" ]]; then
  RESUME_CKPT=$(find "${RUN_DIR}" -maxdepth 1 -type f -name 'ckpt_step*.npz' | sort -V | tail -n 1)
  if [[ -n "${RESUME_CKPT}" ]]; then
    RESUME_STEP=$(basename "${RESUME_CKPT}" .npz)
    RESUME_STEP="${RESUME_STEP#ckpt_step}"
    if (( RESUME_STEP >= MAX_STEPS )); then
      echo "${RUN_NAME} already has ${RESUME_CKPT} at or beyond max step ${MAX_STEPS}; nothing to do."
      exit 0
    fi
    echo "Resuming ${RUN_NAME} from ${RESUME_CKPT} at step ${RESUME_STEP}"
  fi
fi

if [[ -z "${RESUME_CKPT}" && -n "${START_STEP}" ]]; then
  RESUME_CKPT="${START_CKPT}"
  RESUME_STEP="${START_STEP}"
fi

CMD=(
  python -u -m src.models.mamba.training.segments_training
  --model "${MODEL}"
  --data-path "${DATA_PATH}"
  --data-source "${DATA_SOURCE}"
  --prepared-data-root "${PREPARED_DATA_ROOT}"
  --stats-dir "${STATS_DIR}"
  --out-dir "${OUT_DIR}"
  --run-name "${RUN_NAME}"
  --val-year "${VAL_YEAR}"
  --resolution "${RESOLUTION}"
  --mesh-size "${MESH_SIZE}"
  --width "${WIDTH}"
  --processor-msg-steps "${MSG}"
  --input-duration "${INPUT_DURATION}"
  --target-steps 1
  --len-segment "${LEN_SEGMENT}"
  --bptt-steps "${BPTT_STEPS}"
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
  --precision "${PRECISION}"
  --seed "${SEED}"
  --temporal-backbone mamba
  --temporal-location mesh_processor_interleaved
  --temporal-stateful
  --temporal-d-inner "${D_INNER}"
  --temporal-d-state "${D_STATE}"
  --temporal-d-conv 4
  --temporal-dt-rank auto
  --temporal-layers 1
  --temporal-dropout 0.0
  --temporal-insert-count "${INSERT_COUNT}"
)

if [[ -n "${TRAIN_START_YEAR}" && -n "${TRAIN_END_YEAR}" ]]; then
  CMD+=(--train-start-year "${TRAIN_START_YEAR}" --train-end-year "${TRAIN_END_YEAR}")
fi

if [[ "${MODEL}" == "gc_mamba" ]]; then
  if [[ -n "${RESUME_CKPT}" ]]; then
    CMD+=(--ckpt-in "${RESUME_CKPT}" --resume-step "${RESUME_STEP}" --trainable-part "${TRAINABLE_PART}")
    if [[ "${STAGE}" == "frozen" ]]; then
      CMD+=(--zero-init-temporal-out)
    fi
  else
    CMD+=(
      --ckpt-in "${BASE_CKPT}"
      --init-from-graphcast-ckpt "${BASE_CKPT}"
      --trainable-part mamba
      --zero-init-temporal-out
    )
  fi
else
  if [[ -n "${RESUME_CKPT}" ]]; then
    CMD+=(--baseline-ckpt "${BASE_CKPT}" --resume-ckpt "${RESUME_CKPT}" --resume-step "${RESUME_STEP}")
  else
    CMD+=(--baseline-ckpt "${BASE_CKPT}")
  fi
fi

"${CMD[@]}"
