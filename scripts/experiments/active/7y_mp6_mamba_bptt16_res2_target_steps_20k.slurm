#!/bin/bash
#SBATCH --job-name=7y-mp6-r2-tgt-b16
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/7y_mp6_mamba_res2_target_bptt16_%A_%a.out
#SBATCH --error=logs/7y_mp6_mamba_res2_target_bptt16_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
source scripts/graphcast_env.sh

BASE_DIR="artifacts/checkpoints/7_years/vanilla_gc/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k"
TARGET_STEPS_SET=(4 8 12)
D_PAIRS=(
  "64:32"
  "128:64"
)
VARIANTS=(
  "gc_mamba:2"
  "residual_mamba:1"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
COMBOS_PER_TARGET=$((${#D_PAIRS[@]} * ${#VARIANTS[@]}))
EXPECTED_TASKS=$((${#TARGET_STEPS_SET[@]} * COMBOS_PER_TARGET))
if (( TASK_ID < 0 || TASK_ID >= EXPECTED_TASKS )); then
  echo "Task ${TASK_ID} outside expected range 0-$((EXPECTED_TASKS - 1))" >&2
  exit 1
fi

TARGET_IDX=$((TASK_ID / COMBOS_PER_TARGET))
REM=$((TASK_ID % COMBOS_PER_TARGET))
D_PAIR_IDX=$((REM / ${#VARIANTS[@]}))
VARIANT_IDX=$((REM % ${#VARIANTS[@]}))

TARGET_STEPS="${TARGET_STEPS_SET[$TARGET_IDX]}"
IFS=: read -r D_INNER D_STATE <<< "${D_PAIRS[$D_PAIR_IDX]}"
IFS=: read -r MODEL INSERT_COUNT <<< "${VARIANTS[$VARIANT_IDX]}"

CKPT_IN="${BASE_DIR}/ckpt_best.npz"
RUN_CONFIG="${BASE_DIR}/run_config.json"

if [[ ! -f "${CKPT_IN}" ]]; then
  echo "Missing checkpoint: ${CKPT_IN}" >&2
  exit 1
fi
if [[ ! -f "${RUN_CONFIG}" ]]; then
  echo "Missing run config: ${RUN_CONFIG}" >&2
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
OUT_DIR=${OUT_DIR:-artifacts/checkpoints/7_years/mamba_frozen_from_vanilla_mp6_20k_res2_target_steps_bptt16}
MAX_STEPS=${MAX_STEPS:-20000}
LEN_SEGMENT=${LEN_SEGMENT:-32}
BPTT_STEPS=${BPTT_STEPS:-16}
CHUNK_LOAD_WORKERS=${CHUNK_LOAD_WORKERS:-6}
EVAL_NUM_SEGMENTS=${EVAL_NUM_SEGMENTS:-16}
FINAL_EVAL_NUM_SEGMENTS=${FINAL_EVAL_NUM_SEGMENTS:-all}
BATCH_SIZE=${BATCH_SIZE_OVERRIDE:-${BATCH_SIZE}}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE_OVERRIDE:-${EVAL_BATCH_SIZE}}
RESIDUAL_AR_FEEDBACK=${RESIDUAL_AR_FEEDBACK:-baseline_plus_residual}
DRY_RUN=${DRY_RUN:-0}

BASE_NAME="$(basename "${BASE_DIR}")"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"
RUN_NAME="${BASE_NAME}_${MODEL}_tc${INSERT_COUNT}_di${D_INNER}_ds${D_STATE}_20k_target_step${TARGET_STEPS}_bptt16${RUN_NAME_SUFFIX}"
RUN_DIR="${OUT_DIR}/${RUN_NAME}"

echo "Task ${TASK_ID}/${EXPECTED_TASKS}: base=${BASE_NAME} model=${MODEL} insert=${INSERT_COUNT} d_inner=${D_INNER} d_state=${D_STATE} target_steps=${TARGET_STEPS} bptt_steps=${BPTT_STEPS}"
echo "Run directory: ${RUN_DIR}"
if [[ "${MODEL}" == "residual_mamba" ]]; then
  echo "Residual AR feedback: ${RESIDUAL_AR_FEEDBACK}"
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
  --target-steps "${TARGET_STEPS}"
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
  CMD+=(
    --ckpt-in "${CKPT_IN}"
    --init-from-graphcast-ckpt "${CKPT_IN}"
    --trainable-part mamba
    --zero-init-temporal-out
  )
else
  CMD+=(
    --baseline-ckpt "${CKPT_IN}"
    --residual-ar-feedback "${RESIDUAL_AR_FEEDBACK}"
  )
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Resolved command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
