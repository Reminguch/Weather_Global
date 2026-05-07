#!/bin/bash
#SBATCH --job-name=gc-standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gc_standard_%j.out
#SBATCH --error=logs/gc_standard_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
source scripts/graphcast_env.sh

DATA_PATH=${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}
DATA_SOURCE=${DATA_SOURCE:-prepared_array}
PREPARED_DATA_ROOT=${PREPARED_DATA_ROOT:-data/graphcast/graphcast/dataset/prepared_stream}
VAL_YEAR=${VAL_YEAR:-2022}
TRAIN_START_YEAR=${TRAIN_START_YEAR:-}
TRAIN_END_YEAR=${TRAIN_END_YEAR:-}

RESOLUTION=${RESOLUTION:-6}
MESH_SIZE=${MESH_SIZE:-3}
WIDTH=${WIDTH:-256}
MSG=${MSG:-1}

BATCH_SIZE=${BATCH_SIZE:-4}
MAX_STEPS=${MAX_STEPS:-50000}
EVAL_EVERY=${EVAL_EVERY:-2000}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
EVAL_NUM_BATCHES=${EVAL_NUM_BATCHES:-16}
FINAL_EVAL_NUM_BATCHES=${FINAL_EVAL_NUM_BATCHES:-all}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-2000}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
SEED=${SEED:-0}
PRECISION=${PRECISION:-bf16}
BATCH_BUILDER=${BATCH_BUILDER:-prepared_array}

OUT_DIR=${OUT_DIR:-artifacts/checkpoints/graphcast_res${RESOLUTION%.*}_manual}
RES_TAG="res${RESOLUTION%.0}"
RUN_NAME=${RUN_NAME:-${RES_TAG}_m${MESH_SIZE}_w${WIDTH}_mp${MSG}_h6_bs${BATCH_SIZE}}
CKPT_IN=${CKPT_IN:-"data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"}
STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}
AUTO_RESUME=${AUTO_RESUME:-0}

RESUME_RUN_DIR="${OUT_DIR}/${RUN_NAME}"
if [[ "${AUTO_RESUME}" == "1" && -d "${RESUME_RUN_DIR}" ]]; then
  LATEST_CKPT=$(ls -v "${RESUME_RUN_DIR}"/ckpt_step*.npz 2>/dev/null | tail -n 1 || true)
  if [[ -n "${LATEST_CKPT}" ]]; then
    CKPT_IN="${LATEST_CKPT}"
    RESUME_STEP=$(basename "${LATEST_CKPT}" .npz | sed 's/ckpt_step//')
    echo "Resuming from ${LATEST_CKPT} at step ${RESUME_STEP}"
  fi
fi

CMD=(
  python -u -m src.models.graphcast.training.standard_training
  --data-path "${DATA_PATH}"
  --data-source "${DATA_SOURCE}"
  --prepared-data-root "${PREPARED_DATA_ROOT}"
  --val-year "${VAL_YEAR}"
  --resolution "${RESOLUTION}"
  --mesh-size "${MESH_SIZE}"
  --width "${WIDTH}"
  --processor-msg-steps "${MSG}"
  --batch-size "${BATCH_SIZE}"
  --max-steps "${MAX_STEPS}"
  --eval-every "${EVAL_EVERY}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --eval-num-batches "${EVAL_NUM_BATCHES}"
  --final-eval-num-batches "${FINAL_EVAL_NUM_BATCHES}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --seed "${SEED}"
  --precision "${PRECISION}"
  --batch-builder "${BATCH_BUILDER}"
  --out-dir "${OUT_DIR}"
  --run-name "${RUN_NAME}"
  --ckpt-in "${CKPT_IN}"
  --stats-dir "${STATS_DIR}"
)

if [[ -n "${RESUME_STEP:-}" ]]; then
  CMD+=(--resume-step "${RESUME_STEP}")
fi

if [[ -n "${TRAIN_START_YEAR}" && -n "${TRAIN_END_YEAR}" ]]; then
  CMD+=(--train-start-year "${TRAIN_START_YEAR}" --train-end-year "${TRAIN_END_YEAR}")
fi

"${CMD[@]}"
