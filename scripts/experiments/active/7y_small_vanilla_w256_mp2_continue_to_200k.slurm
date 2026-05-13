#!/bin/bash
#SBATCH --job-name=gc7y-200k
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gc7y_w256_mp2_continue200k_%x_%j.out
#SBATCH --error=logs/gc7y_w256_mp2_continue200k_%x_%j.err

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
case "${RES_KEY}" in
  res2)
    RUN_NAME="vanilla_gc_7y_res2_m4_w256_mp2_h6_bs8_accum1_stream200k"
    ;;
  res3)
    RUN_NAME="vanilla_gc_7y_res3_m4_w256_mp2_h6_bs8_accum1_stream200k"
    ;;
  res6)
    RUN_NAME="vanilla_gc_7y_res6_m3_w256_mp2_h6_bs8_accum1_stream200k"
    ;;
  *)
    echo "Unknown RES_KEY=${RES_KEY}; expected res2, res3, or res6." >&2
    exit 1
    ;;
esac

OUT_DIR=${OUT_DIR:-artifacts/checkpoints/7_years/small_experiments}
RUN_DIR="${OUT_DIR}/${RUN_NAME}"
RUN_CONFIG="${RUN_DIR}/run_config.json"
CKPT_IN="${RUN_DIR}/ckpt_step150000.npz"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Missing renamed 200k run directory: ${RUN_DIR}" >&2
  exit 1
fi
if [[ ! -f "${RUN_CONFIG}" ]]; then
  echo "Missing run config: ${RUN_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${CKPT_IN}" ]]; then
  echo "Missing exact 150k resume checkpoint: ${CKPT_IN}" >&2
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
    str(cfg.get("grad_accum_steps", 1)),
    str(cfg["eval_batch_size"]),
    str(cfg.get("eval_every", 2000)),
    str(cfg.get("checkpoint_every", 2000)),
    str(pipeline.get("data_cache_mode", "never")),
    str(pipeline.get("data_cache_max_gib", 48)),
    str(pipeline.get("batch_builder", "prepared_array")),
    str(pipeline.get("prefetch_workers", 4)),
    str(pipeline.get("prefetch_depth", 8)),
    str(pipeline.get("prefetch_device_depth", 1)),
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
GRAD_ACCUM_STEPS="${CFG[12]}"
EVAL_BATCH_SIZE="${CFG[13]}"
EVAL_EVERY="${CFG[14]}"
CHECKPOINT_EVERY="${CFG[15]}"
DATA_CACHE_MODE="${CFG[16]}"
DATA_CACHE_MAX_GIB="${CFG[17]}"
BATCH_BUILDER="${CFG[18]}"
PREFETCH_WORKERS="${CFG[19]}"
PREFETCH_DEPTH="${CFG[20]}"
PREFETCH_DEVICE_DEPTH="${CFG[21]}"
LR="${CFG[22]}"
WEIGHT_DECAY="${CFG[23]}"
PRECISION="${CFG[24]}"
SEED="${CFG[25]}"

STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}
MAX_STEPS=${MAX_STEPS:-200000}
RESUME_STEP=${RESUME_STEP:-150000}
EVAL_NUM_BATCHES=${EVAL_NUM_BATCHES:-16}
FINAL_EVAL_NUM_BATCHES=${FINAL_EVAL_NUM_BATCHES:-all}

if [[ "${RESUME_STEP}" != "150000" ]]; then
  echo "This continuation expects RESUME_STEP=150000, got ${RESUME_STEP}" >&2
  exit 1
fi
if [[ "${MAX_STEPS}" != "200000" ]]; then
  echo "This continuation expects MAX_STEPS=200000, got ${MAX_STEPS}" >&2
  exit 1
fi

echo "Continuing ${RUN_NAME}: ${RESUME_STEP} -> ${MAX_STEPS}"

CMD=(
  python -u -m src.models.graphcast.training.standard_training
  --data-path "${DATA_PATH}"
  --data-source "${DATA_SOURCE}"
  --prepared-data-root "${PREPARED_DATA_ROOT}"
  --stats-dir "${STATS_DIR}"
  --ckpt-in "${CKPT_IN}"
  --resume-step "${RESUME_STEP}"
  --out-dir "${OUT_DIR}"
  --run-name "${RUN_NAME}"
  --val-year "${VAL_YEAR}"
  --resolution "${RESOLUTION}"
  --mesh-size "${MESH_SIZE}"
  --width "${WIDTH}"
  --processor-msg-steps "${MSG}"
  --input-duration "${INPUT_DURATION}"
  --target-steps 1
  --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --eval-num-batches "${EVAL_NUM_BATCHES}"
  --final-eval-num-batches "${FINAL_EVAL_NUM_BATCHES}"
  --data-cache-mode "${DATA_CACHE_MODE}"
  --data-cache-max-gib "${DATA_CACHE_MAX_GIB}"
  --batch-builder "${BATCH_BUILDER}"
  --prefetch-workers "${PREFETCH_WORKERS}"
  --prefetch-depth "${PREFETCH_DEPTH}"
  --prefetch-device-depth "${PREFETCH_DEVICE_DEPTH}"
  --max-steps "${MAX_STEPS}"
  --eval-every "${EVAL_EVERY}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --precision "${PRECISION}"
  --seed "${SEED}"
)

if [[ -n "${TRAIN_START_YEAR}" && -n "${TRAIN_END_YEAR}" ]]; then
  CMD+=(--train-start-year "${TRAIN_START_YEAR}" --train-end-year "${TRAIN_END_YEAR}")
fi

"${CMD[@]}"
