#!/bin/bash
#SBATCH --job-name=gc-stream-frozen-resmamba
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/gc_stream_frozen_resmamba_%A_%a.out
#SBATCH --error=logs/gc_stream_frozen_resmamba_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
source scripts/graphcast_env.sh

STREAM_ROOTS=(
  "artifacts/checkpoints/graphcast_res2_stream"
  "artifacts/checkpoints/graphcast_res4_stream"
  "artifacts/checkpoints/graphcast_res6_stream"
  "artifacts/checkpoints/graphcast_res15_stream"
)
D_INNER_VALUES=(${D_INNER_VALUES:-16 32})
BASELINE_SUFFIX=${BASELINE_SUFFIX:-w1024_mp2_h6_bs4}

mapfile -t BASELINE_RUN_DIRS < <(
  for root in "${STREAM_ROOTS[@]}"; do
    find "${root}" -mindepth 1 -maxdepth 1 -type d -name "*_${BASELINE_SUFFIX}"
  done | sort
)

if (( ${#BASELINE_RUN_DIRS[@]} == 0 )); then
  echo "No baseline run directories found under: ${STREAM_ROOTS[*]} for suffix ${BASELINE_SUFFIX}"
  exit 1
fi
if (( ${#D_INNER_VALUES[@]} == 0 )); then
  echo "D_INNER_VALUES resolved to an empty list."
  exit 1
fi

TOTAL_COMBOS=$(( ${#BASELINE_RUN_DIRS[@]} * ${#D_INNER_VALUES[@]} ))
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= TOTAL_COMBOS )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0-$(( TOTAL_COMBOS - 1 ))"
  echo "You can also narrow the job with sbatch --array=..."
  exit 1
fi

RUN_IDX=$(( TASK_ID / ${#D_INNER_VALUES[@]} ))
D_INNER_IDX=$(( TASK_ID % ${#D_INNER_VALUES[@]} ))
RUN_DIR="${BASELINE_RUN_DIRS[${RUN_IDX}]}"
D_INNER="${D_INNER_VALUES[${D_INNER_IDX}]}"
RUN_CONFIG="${RUN_DIR}/run_config.json"

if [[ ! -f "${RUN_CONFIG}" ]]; then
  echo "Missing run config: ${RUN_CONFIG}"
  exit 1
fi

if [[ -f "${RUN_DIR}/ckpt_best.npz" ]]; then
  BASELINE_CKPT="${RUN_DIR}/ckpt_best.npz"
else
  BASELINE_CKPT="$(find "${RUN_DIR}" -maxdepth 1 -type f -name 'ckpt_step*.npz' | sort -V | tail -n 1)"
fi

if [[ -z "${BASELINE_CKPT}" || ! -f "${BASELINE_CKPT}" ]]; then
  echo "Could not find baseline checkpoint in ${RUN_DIR}"
  exit 1
fi

mapfile -t CFG_VALUES < <(
  python - "${RUN_CONFIG}" "${RUN_DIR}" <<'PY'
import json
import pathlib
import sys

cfg_path = pathlib.Path(sys.argv[1])
run_dir = pathlib.Path(sys.argv[2])
cfg = json.loads(cfg_path.read_text())
model = cfg["model_config"]
task = cfg["task_config"]

values = [
    run_dir.name,
    str(cfg["data_path"]),
    str(model["resolution"]),
    str(model["mesh_size"]),
    str(model["latent_size"]),
    str(model["gnn_msg_steps"]),
    str(cfg["val_year"]),
    "" if cfg.get("train_start_year") is None else str(cfg["train_start_year"]),
    "" if cfg.get("train_end_year") is None else str(cfg["train_end_year"]),
    str(cfg["batch_size"]),
    str(cfg["eval_batch_size"]),
    str(cfg["max_steps"]),
    str(cfg["eval_every"]),
    str(cfg["checkpoint_every"]),
    str(cfg["lr"]),
    str(cfg["weight_decay"]),
    str(cfg["precision"]),
    str(task["input_duration"]),
]
for value in values:
    print(value)
PY
)

BASE_RUN_NAME="${CFG_VALUES[0]}"
DATA_PATH="${DATA_PATH:-${CFG_VALUES[1]}}"
RESOLUTION="${RESOLUTION:-${CFG_VALUES[2]}}"
MESH_SIZE="${MESH_SIZE:-${CFG_VALUES[3]}}"
WIDTH="${WIDTH:-${CFG_VALUES[4]}}"
MSG="${MSG:-${CFG_VALUES[5]}}"
VAL_YEAR="${VAL_YEAR:-${CFG_VALUES[6]}}"
TRAIN_START_YEAR="${TRAIN_START_YEAR:-${CFG_VALUES[7]}}"
TRAIN_END_YEAR="${TRAIN_END_YEAR:-${CFG_VALUES[8]}}"
BATCH_SIZE="${BATCH_SIZE:-${CFG_VALUES[9]}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${CFG_VALUES[10]}}"
MAX_STEPS="${MAX_STEPS:-${CFG_VALUES[11]}}"
EVAL_EVERY="${EVAL_EVERY:-${CFG_VALUES[12]}}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-${CFG_VALUES[13]}}"
LR="${LR:-${CFG_VALUES[14]}}"
WEIGHT_DECAY="${WEIGHT_DECAY:-${CFG_VALUES[15]}}"
PRECISION="${PRECISION:-${CFG_VALUES[16]}}"
INPUT_DURATION="${INPUT_DURATION:-${CFG_VALUES[17]}}"

STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}
OUT_DIR=${OUT_DIR:-artifacts/checkpoints/graphcast_stream_frozen_residual_mamba}

TARGET_STEPS=${TARGET_STEPS:-1}
LEN_SEGMENT=${LEN_SEGMENT:-30}
BPTT_STEPS=${BPTT_STEPS:-6}
CHUNK_LOAD_WORKERS=${CHUNK_LOAD_WORKERS:-6}
DATA_CACHE_MODE=${DATA_CACHE_MODE:-auto}
DATA_CACHE_MAX_GIB=${DATA_CACHE_MAX_GIB:-48}
BATCH_BUILDER=${BATCH_BUILDER:-numpy}
SEED=${SEED:-0}

TEMPORAL_HIDDEN_SIZE=${TEMPORAL_HIDDEN_SIZE:-128}
TEMPORAL_D_STATE=${TEMPORAL_D_STATE:-1}
TEMPORAL_D_CONV=${TEMPORAL_D_CONV:-1}
TEMPORAL_DT_RANK=${TEMPORAL_DT_RANK:-auto}
TEMPORAL_DROPOUT=${TEMPORAL_DROPOUT:-0.0}

RUN_NAME="residual_mamba_int_${BASE_RUN_NAME}_frozen_di${D_INNER}_ds${TEMPORAL_D_STATE}_dc${TEMPORAL_D_CONV}_ls${LEN_SEGMENT}_bptt${BPTT_STEPS}_t${TARGET_STEPS}"

echo "host: $(hostname)"
python --version
which python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "baseline run: ${BASE_RUN_NAME}"
echo "baseline ckpt: ${BASELINE_CKPT}"
echo "baseline cfg: data=${DATA_PATH} res=${RESOLUTION} mesh=${MESH_SIZE} width=${WIDTH} mp=${MSG} val_year=${VAL_YEAR}"
echo "residual mamba: hidden=${TEMPORAL_HIDDEN_SIZE} d_inner=${D_INNER} d_state=${TEMPORAL_D_STATE} d_conv=${TEMPORAL_D_CONV}"
echo "segments: len_segment=${LEN_SEGMENT} bptt_steps=${BPTT_STEPS}"
echo "run: ${OUT_DIR}/${RUN_NAME}"

CMD=(
  python -u scripts/training/residual_memory/train_graphcast_residual_segments.py
  --data-path "${DATA_PATH}"
  --stats-dir "${STATS_DIR}"
  --baseline-ckpt "${BASELINE_CKPT}"
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
  --training-target residual
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --precision "${PRECISION}"
  --seed "${SEED}"
  --temporal-backbone mamba
  --temporal-location mesh_processor_interleaved
  --temporal-stateful
  --temporal-hidden-size "${TEMPORAL_HIDDEN_SIZE}"
  --temporal-layers 1
  --temporal-d-inner "${D_INNER}"
  --temporal-d-state "${TEMPORAL_D_STATE}"
  --temporal-d-conv "${TEMPORAL_D_CONV}"
  --temporal-dt-rank "${TEMPORAL_DT_RANK}"
  --temporal-dropout "${TEMPORAL_DROPOUT}"
)

if [[ -n "${TRAIN_START_YEAR}" && -n "${TRAIN_END_YEAR}" ]]; then
  CMD+=(--train-start-year "${TRAIN_START_YEAR}" --train-end-year "${TRAIN_END_YEAR}")
fi

"${CMD[@]}"
