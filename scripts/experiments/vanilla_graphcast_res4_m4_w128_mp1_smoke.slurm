#!/bin/bash
#SBATCH --job-name=gc-r4-smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/gc_r4_smoke_%j.out
#SBATCH --error=logs/gc_r4_smoke_%j.err

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
VAL_YEAR=${VAL_YEAR:-2022}
TRAIN_START_YEAR=${TRAIN_START_YEAR:-}
TRAIN_END_YEAR=${TRAIN_END_YEAR:-}

RESOLUTION=4
MESH_SIZE=4
WIDTH=128
MSG=1

BATCH_SIZE=${BATCH_SIZE:-4}
MAX_STEPS=2000
EVAL_EVERY=500
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-500}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
SEED=${SEED:-0}
PRECISION=${PRECISION:-bf16}

OUT_DIR=${OUT_DIR:-artifacts/checkpoints/experiments_smoke_graphcast_res4}
RUN_NAME=${RUN_NAME:-vanilla_graphcast_res4_m4_w128_mp1_smoke}
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
  --val-year "${VAL_YEAR}"
  --resolution "${RESOLUTION}"
  --mesh-size "${MESH_SIZE}"
  --width "${WIDTH}"
  --processor-msg-steps "${MSG}"
  --batch-size "${BATCH_SIZE}"
  --max-steps "${MAX_STEPS}"
  --eval-every "${EVAL_EVERY}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --seed "${SEED}"
  --precision "${PRECISION}"
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
