#!/bin/bash
#SBATCH --job-name=gcm-from-gc
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-13
#SBATCH --output=logs/gcm_from_gc_%A_%a.out
#SBATCH --error=logs/gcm_from_gc_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
source scripts/graphcast_env.sh

BASE_DIRS=(
  "artifacts/checkpoints/pilot_graphcast_segments_ms_sweep/base_seg_res2_m4_w128_mp1_ls30_bptt6_t1"
  "artifacts/checkpoints/pilot_graphcast_segments_ms_sweep/base_seg_res2_m4_w128_mp2_ls30_bptt6_t1"
  "artifacts/checkpoints/pilot_graphcast_segments_ms_sweep/base_seg_res2_m4_w128_mp3_ls30_bptt6_t1"
  "artifacts/checkpoints/pilot_graphcast_segments_ms_sweep/base_seg_res2_m4_w256_mp1_ls30_bptt6_t1"
  "artifacts/checkpoints/pilot_graphcast_segments_ms_sweep/base_seg_res4_m3_w128_mp1_ls30_bptt6_t1"
  "artifacts/checkpoints/pilot_graphcast_segments_ms_sweep/base_seg_res9_m2_w128_mp1_ls30_bptt6_t1"
  "artifacts/checkpoints/pilot_graphcast_segments_ms_sweep/base_seg_res15_m2_w128_mp1_ls30_bptt6_t1"
)
D_INNERS=(32 64)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
BASE_IDX=$((TASK_ID / ${#D_INNERS[@]}))
D_IDX=$((TASK_ID % ${#D_INNERS[@]}))

BASE_DIR="${BASE_DIRS[$BASE_IDX]}"
D_INNER="${D_INNERS[$D_IDX]}"
CKPT_IN="${BASE_DIR}/ckpt_best.npz"
RUN_CONFIG="${BASE_DIR}/run_config.json"

if [[ ! -f "${CKPT_IN}" ]]; then
  echo "Missing checkpoint: ${CKPT_IN}"
  exit 1
fi
if [[ ! -f "${RUN_CONFIG}" ]]; then
  echo "Missing run config: ${RUN_CONFIG}"
  exit 1
fi

readarray -t CFG < <(python - "${RUN_CONFIG}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)

model = cfg["model_config"]
task = cfg["task_config"]
segment = cfg["segment_training"]
pipeline = cfg["data_pipeline"]
fields = [
    cfg["data_path"],
    str(cfg["val_year"]),
    "" if cfg.get("train_start_year") is None else str(cfg["train_start_year"]),
    "" if cfg.get("train_end_year") is None else str(cfg["train_end_year"]),
    str(model["resolution"]),
    str(model["mesh_size"]),
    str(model["latent_size"]),
    str(model["gnn_msg_steps"]),
    str(task["input_duration"]),
    str(segment["len_segment"]),
    str(segment["bptt_steps"]),
    str(segment["chunk_load_workers"]),
    str(cfg["batch_size"]),
    str(cfg["eval_batch_size"]),
    str(cfg["max_steps"]),
    str(cfg["eval_every"]),
    str(cfg["checkpoint_every"]),
    str(pipeline["data_cache_mode"]),
    str(pipeline["data_cache_max_gib"]),
    str(pipeline["batch_builder"]),
    str(cfg["lr"]),
    str(cfg["weight_decay"]),
    str(cfg["precision"]),
    str(cfg["seed"]),
]
print("\n".join(fields))
PY
)

DATA_PATH="${CFG[0]}"
VAL_YEAR="${CFG[1]}"
TRAIN_START_YEAR="${CFG[2]}"
TRAIN_END_YEAR="${CFG[3]}"
RESOLUTION="${CFG[4]}"
MESH_SIZE="${CFG[5]}"
WIDTH="${CFG[6]}"
MSG="${CFG[7]}"
INPUT_DURATION="${CFG[8]}"
LEN_SEGMENT="${CFG[9]}"
BPTT_STEPS="${CFG[10]}"
CHUNK_LOAD_WORKERS="${CFG[11]}"
BATCH_SIZE="${CFG[12]}"
EVAL_BATCH_SIZE="${CFG[13]}"
MAX_STEPS="${CFG[14]}"
EVAL_EVERY="${CFG[15]}"
CHECKPOINT_EVERY="${CFG[16]}"
DATA_CACHE_MODE="${CFG[17]}"
DATA_CACHE_MAX_GIB="${CFG[18]}"
BATCH_BUILDER="${CFG[19]}"
LR="${CFG[20]}"
WEIGHT_DECAY="${CFG[21]}"
PRECISION="${CFG[22]}"
SEED="${CFG[23]}"

STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}
OUT_DIR=${OUT_DIR:-artifacts/checkpoints/gc_mamba_from_vanilla_ms_sweep}
RUN_NAME="$(basename "${BASE_DIR}")_mamba_di${D_INNER}_ds1"

CMD=(
  python -u -m src.models.mamba.training.segments_training
  --model gc_mamba
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
  --target-steps 1
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
  --temporal-backbone mamba
  --temporal-location mesh_processor_interleaved
  --temporal-stateful
  --temporal-hidden-size "${WIDTH}"
  --temporal-d-inner "${D_INNER}"
  --temporal-d-state 1
  --temporal-d-conv 4
  --temporal-dt-rank auto
  --temporal-layers 1
  --temporal-dropout 0.0
  --zero-init-temporal-out
  --trainable-part mamba
  --ckpt-in "${CKPT_IN}"
  --init-from-graphcast-ckpt "${CKPT_IN}"
)

if [[ -n "${TRAIN_START_YEAR}" && -n "${TRAIN_END_YEAR}" ]]; then
  CMD+=(--train-start-year "${TRAIN_START_YEAR}" --train-end-year "${TRAIN_END_YEAR}")
fi

"${CMD[@]}"
