#!/bin/bash
#SBATCH --job-name=7y-mp6-r2-ds16-k12-glr
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=07:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/7y_mp6_gc_mamba_ds16_k12_grouped_lr_release20k_%A_%a.out
#SBATCH --error=logs/7y_mp6_gc_mamba_ds16_k12_grouped_lr_release20k_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
source scripts/graphcast_env.sh

SOURCE_ROOT=${SOURCE_ROOT:-artifacts/checkpoints/7_years/mamba_frozen_from_vanilla_mp6_20k_res2_target_steps_bptt16}
OUT_DIR=${OUT_DIR:-artifacts/checkpoints/7_years/mamba_release_all_from_frozen20k_res2_ds16_k12_gclr1e6_mlr1e5_20k}
BASE_NAME="vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k"
TARGET_STEPS_SET=(12)
D_INNERS=(16 64)
D_STATE=16
MODEL="gc_mamba"
INSERT_COUNT=2
RELEASE_SUFFIX="_release_all20k_gclr1e6_mlr1e5"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
COMBOS_PER_TARGET=${#D_INNERS[@]}
EXPECTED_TASKS=$((${#TARGET_STEPS_SET[@]} * COMBOS_PER_TARGET))
if (( TASK_ID < 0 || TASK_ID >= EXPECTED_TASKS )); then
  echo "Task ${TASK_ID} outside expected range 0-$((EXPECTED_TASKS - 1))" >&2
  exit 1
fi

TARGET_IDX=$((TASK_ID / COMBOS_PER_TARGET))
D_INNER_IDX=$((TASK_ID % COMBOS_PER_TARGET))

TARGET_STEPS="${TARGET_STEPS_SET[$TARGET_IDX]}"
D_INNER="${D_INNERS[$D_INNER_IDX]}"
SOURCE_RUN="${BASE_NAME}_${MODEL}_tc${INSERT_COUNT}_di${D_INNER}_ds${D_STATE}_20k_target_step${TARGET_STEPS}_bptt16"
SOURCE_RUN_DIR="${SOURCE_ROOT}/${SOURCE_RUN}"
SOURCE_CKPT="${SOURCE_RUN_DIR}/ckpt_step20000.npz"
RUN_CONFIG="${SOURCE_RUN_DIR}/run_config.json"
RUN_NAME="${SOURCE_RUN}${RELEASE_SUFFIX}"
RUN_DIR="${OUT_DIR}/${RUN_NAME}"

if [[ ! -f "${SOURCE_CKPT}" ]]; then
  echo "Missing source checkpoint: ${SOURCE_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${RUN_CONFIG}" ]]; then
  echo "Missing source run config: ${RUN_CONFIG}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
cat > "${OUT_DIR}/README.md" <<EOF
# Grouped-LR k12 Release Diagnostic Run

Hypothesis: constrain GraphCast updates with a smaller LR while letting Mamba continue temporal-history learning during release.

Source checkpoints:
\`${SOURCE_ROOT}\`

Release setup:
- trainable_part: all
- graphcast_lr: 1e-6
- mamba_lr: 1e-5
- base lr flag: 1e-5
- target_steps: 12
- bptt_steps: 16
- d_state: 16
- variants: di16/ds16/k12 and di64/ds16/k12

Diagnostics:
- Each run writes \`ar_gradient_alignment.jsonl\`.
- Rows decompose the AR-tail objective into early horizons 1-4, mid horizons 5-8, and late horizons 9-12.
- For each step/chunk, the file records losses, gradient norms, LR-weighted update norms, dots, and cosines to the late gradient for \`all\`, \`graphcast\`, and \`mamba\` parameter groups.
- Diagnostics are passive and do not change training gradients, parameters, optimizer state, or loss weighting.

Expected eval comparison:
- experiment: \`res2_ds16_gc_mamba_k12_grouped_lr_release_warm_leads1_9d\`
- data: \`plots/analyze_models/data/resolution_eval/res2_ds16_gc_mamba_k12_grouped_lr_release_warm_leads1_9d\`
- images: \`plots/analyze_models/images/resolution_eval/res2_ds16_gc_mamba_k12_grouped_lr_release_warm_leads1_9d\`

di128/ds16 is intentionally skipped because it is absent from the named source root.
EOF

readarray -t CFG < <(python - "${RUN_CONFIG}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)

model = cfg["model_config"]
task = cfg["task_config"]
pipeline = cfg["data_pipeline"]
temporal = cfg["temporal_config"]
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
    str(cfg.get("eval_every", 2000)),
    str(cfg.get("checkpoint_every", 2000)),
    str(pipeline.get("data_cache_mode", "never")),
    str(pipeline.get("data_cache_max_gib", 48)),
    str(pipeline.get("batch_builder", "prepared_array")),
    str(cfg.get("precision", "bf16")),
    str(cfg.get("seed", 0)),
    str(temporal.get("d_conv", 4)),
    str(temporal.get("dt_rank", "auto")),
    str(temporal.get("layers", 1)),
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
EVAL_EVERY="${EVAL_EVERY:-${CFG[11]}}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-${CFG[12]}}"
DATA_CACHE_MODE="${DATA_CACHE_MODE:-${CFG[13]}}"
DATA_CACHE_MAX_GIB="${DATA_CACHE_MAX_GIB:-${CFG[14]}}"
BATCH_BUILDER="${BATCH_BUILDER:-${CFG[15]}}"
PRECISION="${PRECISION:-${CFG[16]}}"
SEED="${SEED:-${CFG[17]}}"
TEMPORAL_D_CONV="${TEMPORAL_D_CONV:-${CFG[18]}}"
TEMPORAL_DT_RANK="${TEMPORAL_DT_RANK:-${CFG[19]}}"
TEMPORAL_LAYERS="${TEMPORAL_LAYERS:-${CFG[20]}}"

STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}
MAX_STEPS=${MAX_STEPS:-40000}
RESUME_STEP=${RESUME_STEP:-20000}
LR=${LR:-1e-5}
GRAPHCAST_LR=${GRAPHCAST_LR:-1e-6}
MAMBA_LR=${MAMBA_LR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
LEN_SEGMENT=${LEN_SEGMENT:-32}
BPTT_STEPS=${BPTT_STEPS:-16}
CHUNK_LOAD_WORKERS=${CHUNK_LOAD_WORKERS:-6}
EVAL_NUM_SEGMENTS=${EVAL_NUM_SEGMENTS:-16}
FINAL_EVAL_NUM_SEGMENTS=${FINAL_EVAL_NUM_SEGMENTS:-all}
BATCH_SIZE=${BATCH_SIZE_OVERRIDE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE_OVERRIDE:-1}
AR_LOSS_MODE=${AR_LOSS_MODE:-tail_uniform}
AR_GRADIENT_ALIGNMENT_EVERY=${AR_GRADIENT_ALIGNMENT_EVERY:-${EVAL_EVERY}}
AR_GRADIENT_ALIGNMENT_NUM_CHUNKS=${AR_GRADIENT_ALIGNMENT_NUM_CHUNKS:-1}
MEMORY_MODE=${MEMORY_MODE:-standard}
RESUME_FROM_LATEST=${RESUME_FROM_LATEST:-1}
DRY_RUN=${DRY_RUN:-0}

CKPT_IN="${SOURCE_CKPT}"
if [[ "${RESUME_FROM_LATEST}" == "1" && -d "${RUN_DIR}" ]]; then
  LATEST_CKPT=$(find "${RUN_DIR}" -maxdepth 1 -type f -name 'ckpt_step*.npz' | sort -V | tail -n 1)
  if [[ -n "${LATEST_CKPT}" ]]; then
    LATEST_STEP=$(basename "${LATEST_CKPT}" .npz)
    LATEST_STEP="${LATEST_STEP#ckpt_step}"
    if (( LATEST_STEP >= MAX_STEPS )); then
      echo "${RUN_NAME} already has ${LATEST_CKPT} at or beyond max step ${MAX_STEPS}; nothing to do."
      exit 0
    fi
    CKPT_IN="${LATEST_CKPT}"
    RESUME_STEP="${LATEST_STEP}"
    echo "Resuming existing release run from ${CKPT_IN} at step ${RESUME_STEP}"
  fi
fi

echo "Task ${TASK_ID}/${EXPECTED_TASKS}: source=${SOURCE_RUN} target_steps=${TARGET_STEPS} d_inner=${D_INNER} d_state=${D_STATE}"
echo "Release run directory: ${RUN_DIR}"
echo "Start checkpoint: ${CKPT_IN}"
echo "Resume step: ${RESUME_STEP}; max steps: ${MAX_STEPS}; trainable_part=all"
echo "LRs: graphcast=${GRAPHCAST_LR}; mamba=${MAMBA_LR}; base_lr=${LR}; weight_decay=${WEIGHT_DECAY}"
echo "AR gradient alignment: every=${AR_GRADIENT_ALIGNMENT_EVERY}; chunks=${AR_GRADIENT_ALIGNMENT_NUM_CHUNKS}"

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
  --autoregressive-loss-mode "${AR_LOSS_MODE}"
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
  --graphcast-lr "${GRAPHCAST_LR}"
  --mamba-lr "${MAMBA_LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --precision "${PRECISION}"
  --seed "${SEED}"
  --temporal-backbone mamba
  --temporal-location mesh_processor_interleaved
  --temporal-stateful
  --temporal-d-inner "${D_INNER}"
  --temporal-d-state "${D_STATE}"
  --temporal-d-conv "${TEMPORAL_D_CONV}"
  --temporal-dt-rank "${TEMPORAL_DT_RANK}"
  --temporal-layers "${TEMPORAL_LAYERS}"
  --temporal-dropout 0.0
  --temporal-insert-count "${INSERT_COUNT}"
  --ckpt-in "${CKPT_IN}"
  --resume-step "${RESUME_STEP}"
  --trainable-part all
  --memory-mode "${MEMORY_MODE}"
  --ar-gradient-alignment-diagnostics
  --ar-gradient-alignment-every "${AR_GRADIENT_ALIGNMENT_EVERY}"
  --ar-gradient-alignment-num-chunks "${AR_GRADIENT_ALIGNMENT_NUM_CHUNKS}"
)

if [[ -n "${TRAIN_START_YEAR}" && -n "${TRAIN_END_YEAR}" ]]; then
  CMD+=(--train-start-year "${TRAIN_START_YEAR}" --train-end-year "${TRAIN_END_YEAR}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Resolved command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
