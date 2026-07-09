#!/bin/bash
#SBATCH --job-name=gc-res2-sweep
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/gc_stream_%A_%a.out
#SBATCH --error=logs/gc_stream_%A_%a.err

set -euo pipefail

RESOLUTION=${RESOLUTION:-8}

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs

source scripts/graphcast_env.sh

echo "host: $(hostname)"
python --version
which python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Sweep mapping: (WIDTH, MSG)
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    WIDTH=128
    MSG=1
    ;;
  1)
    WIDTH=256
    MSG=1
    ;;
  2)
    WIDTH=512
    MSG=1
    ;;
  3)
    WIDTH=1024
    MSG=1
    ;;
  4)
    WIDTH=128
    MSG=2
    ;;
  5)
    WIDTH=256
    MSG=2
    ;;
  6)
    WIDTH=512
    MSG=2
    ;;
  7)
    WIDTH=1024
    MSG=2
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

DATA_PATH=${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}
VAL_YEAR=${VAL_YEAR:-2022}
TRAIN_START_YEAR=${TRAIN_START_YEAR:-}
TRAIN_END_YEAR=${TRAIN_END_YEAR:-}
# RESOLUTION=${RESOLUTION:-1}
MESH_SIZE=${MESH_SIZE:-3}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_STEPS=${MAX_STEPS:-50000}
EVAL_EVERY=${EVAL_EVERY:-2000}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-2000}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
SEED=${SEED:-0}
PRECISION=${PRECISION:-bf16}
OUT_DIR=${OUT_DIR:-artifacts/checkpoints/graphcast_res${RESOLUTION%.*}_stream}
# Run name includes resolution tag (res2 / res4) from RESOLUTION to match artifacts structure
RES_TAG="res${RESOLUTION%.0}"
RUN_NAME=${RUN_NAME:-${RES_TAG}_m4_w${WIDTH}_mp${MSG}_h6_bs${BATCH_SIZE}}
CKPT_IN=${CKPT_IN:-"data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"}
STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}
AUTO_RESUME=${AUTO_RESUME:-0}

# For every job: resume from latest checkpoint in this run's dir if present (any resolution)
RESUME_RUN_DIR="${OUT_DIR}/${RUN_NAME}"
if [[ "${AUTO_RESUME}" == "1" ]]; then
  if [[ -d "${RESUME_RUN_DIR}" ]]; then
    LATEST_CKPT=$(ls -v "${RESUME_RUN_DIR}"/ckpt_step*.npz 2>/dev/null | tail -n 1)
    if [[ -n "${LATEST_CKPT}" ]]; then
      CKPT_IN="${LATEST_CKPT}"
      RESUME_STEP=$(basename "${LATEST_CKPT}" .npz | sed 's/ckpt_step//')
      echo "Resuming from ${LATEST_CKPT} at step ${RESUME_STEP}"
    fi
  fi
else
  echo "AUTO_RESUME=${AUTO_RESUME}; starting fresh (no --resume-step)."
fi

echo "resolved combo: WIDTH=${WIDTH} MSG=${MSG} RUN_NAME=${RUN_NAME}"

echo "command: python -u scripts/training/train_graphcast.py ..."
CMD=(
  python -u scripts/training/train_graphcast.py
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

# Examples:
#   sbatch scripts/training/train_graphcast.slurm
#   sbatch --export=MAX_STEPS=2000 scripts/training/train_graphcast.slurm
#   sbatch --export=BATCH_SIZE=2,PRECISION=fp32 scripts/training/train_graphcast.slurm
#   sbatch --export=DATA_PATH=data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr scripts/training/train_graphcast.slurm
