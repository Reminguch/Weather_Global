#!/bin/bash
# Continue the staged run in place with canonical global step numbering.
#SBATCH --job-name=gc7y-res2-500k
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gc7y_res2_stream150k_500k_%j.out
#SBATCH --error=logs/gc7y_res2_stream150k_500k_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
source scripts/graphcast_env.sh

RUN_DIR="artifacts/checkpoints/7_years/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream150k"
CKPT_IN="${RUN_DIR}/ckpt_best.npz"
RUN_CONFIG="${RUN_DIR}/run_config.json"
# Actual lineage: 50k initial stage + 20k continuation + 100k current stage.
RESUME_STEP=170000
MAX_STEPS=500000

for required in "${RUN_DIR}" "${CKPT_IN}" "${RUN_CONFIG}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required continuation input: ${required}" >&2
    exit 1
  fi
done

python scripts/training/prepare_global_step_history.py \
  --target "${RUN_DIR}" \
  --stage 'initial_50k|artifacts/checkpoints/7_years/vanilla_gc/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k|0' \
  --stage 'continuation_20k|artifacts/checkpoints/7_years/vanilla_gc_mp6_continue20k/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_continue20k|50000' \
  --stage 'stream150k_local_100k|artifacts/checkpoints/7_years/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream150k|70000'

# The trainer rewrites run_config.json at startup; retain the fully detailed
# config used by the earlier stages for future continuation provenance.
if [[ ! -f "${RUN_DIR}/run_config_before_global_continuation.json" ]]; then
  cp -p "${RUN_CONFIG}" "${RUN_DIR}/run_config_before_global_continuation.json"
fi

readarray -t CFG < <(python - "${RUN_CONFIG}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
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
    str(model["resolution"]), str(model["mesh_size"]),
    str(model["latent_size"]), str(model["gnn_msg_steps"]),
    str(task["input_duration"]), str(cfg["batch_size"]),
    str(cfg.get("grad_accum_steps", 1)), str(cfg["eval_batch_size"]),
    str(cfg.get("eval_every", 2000)), str(cfg.get("checkpoint_every", 2000)),
    str(pipeline.get("data_cache_mode", "never")),
    str(pipeline.get("data_cache_max_gib", 48)),
    str(pipeline.get("batch_builder", "prepared_array")),
    str(pipeline.get("prefetch_workers", 4)),
    str(pipeline.get("prefetch_depth", 8)),
    str(pipeline.get("prefetch_device_depth", 1)),
    str(cfg.get("lr", 1e-4)), str(cfg.get("weight_decay", 1e-4)),
    str(cfg.get("precision", "bf16")), str(cfg.get("seed", 0)),
]
print("\n".join(fields))
PY
)

DATA_PATH="${CFG[0]}"; DATA_SOURCE="${CFG[1]}"; PREPARED_DATA_ROOT="${CFG[2]}"
VAL_YEAR="${CFG[3]}"; TRAIN_START_YEAR="${CFG[4]}"; TRAIN_END_YEAR="${CFG[5]}"
RESOLUTION="${CFG[6]}"; MESH_SIZE="${CFG[7]}"; WIDTH="${CFG[8]}"; MSG="${CFG[9]}"
INPUT_DURATION="${CFG[10]}"; BATCH_SIZE="${CFG[11]}"; GRAD_ACCUM_STEPS="${CFG[12]}"
EVAL_BATCH_SIZE="${CFG[13]}"; EVAL_EVERY="${CFG[14]}"; CHECKPOINT_EVERY="${CFG[15]}"
DATA_CACHE_MODE="${CFG[16]}"; DATA_CACHE_MAX_GIB="${CFG[17]}"; BATCH_BUILDER="${CFG[18]}"
PREFETCH_WORKERS="${CFG[19]}"; PREFETCH_DEPTH="${CFG[20]}"; PREFETCH_DEVICE_DEPTH="${CFG[21]}"
LR="${CFG[22]}"; WEIGHT_DECAY="${CFG[23]}"; PRECISION="${CFG[24]}"; SEED="${CFG[25]}"

echo "Continuing $(basename "${RUN_DIR}"): ${RESUME_STEP} -> ${MAX_STEPS}"

CMD=(
  python -u -m src.models.graphcast.training.standard_training
  --data-path "${DATA_PATH}" --data-source "${DATA_SOURCE}"
  --prepared-data-root "${PREPARED_DATA_ROOT}"
  --stats-dir data/graphcast/graphcast/stats
  --ckpt-in "${CKPT_IN}" --resume-step "${RESUME_STEP}"
  --out-dir "$(dirname "${RUN_DIR}")" --run-name "$(basename "${RUN_DIR}")"
  --val-year "${VAL_YEAR}" --resolution "${RESOLUTION}" --mesh-size "${MESH_SIZE}"
  --width "${WIDTH}" --processor-msg-steps "${MSG}" --input-duration "${INPUT_DURATION}"
  --target-steps 1 --batch-size "${BATCH_SIZE}" --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --eval-batch-size "${EVAL_BATCH_SIZE}" --eval-num-batches 16 --final-eval-num-batches all
  --data-cache-mode "${DATA_CACHE_MODE}" --data-cache-max-gib "${DATA_CACHE_MAX_GIB}"
  --batch-builder "${BATCH_BUILDER}" --prefetch-workers "${PREFETCH_WORKERS}"
  --prefetch-depth "${PREFETCH_DEPTH}" --prefetch-device-depth "${PREFETCH_DEVICE_DEPTH}"
  --max-steps "${MAX_STEPS}" --eval-every "${EVAL_EVERY}" --checkpoint-every "${CHECKPOINT_EVERY}"
  --lr "${LR}" --weight-decay "${WEIGHT_DECAY}" --precision "${PRECISION}" --seed "${SEED}"
)

if [[ -n "${TRAIN_START_YEAR}" && -n "${TRAIN_END_YEAR}" ]]; then
  CMD+=(--train-start-year "${TRAIN_START_YEAR}" --train-end-year "${TRAIN_END_YEAR}")
fi

"${CMD[@]}"
