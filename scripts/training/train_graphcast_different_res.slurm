#!/bin/bash
#SBATCH --job-name=gc-res-sweep
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-47
#SBATCH --output=logs/gc_stream_%A_%a.out
#SBATCH --error=logs/gc_stream_%A_%a.err

set -euo pipefail

# Adaptive sweep script:
# - Sweeps resolutions: 6, 9, 12, 15, 18, 30
# - Uses resolution-dependent mesh mapping for fair comparison
# For a single manual run, use: scripts/training/train_graphcast_standard.slurm

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

# Sweep mapping:
#   6 resolutions x 8 (WIDTH, MSG) combos = 48 jobs total.
#   resolution-major ordering:
#     task_id = resolution_index * 8 + combo_index
RESOLUTIONS=(6 9 12 15 18 30)
COMBOS_PER_RES=8
MAX_TASK_ID=$(( ${#RESOLUTIONS[@]} * COMBOS_PER_RES - 1 ))

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > MAX_TASK_ID )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; expected [0, ${MAX_TASK_ID}]"
  exit 1
fi

RESOLUTION_INDEX=$(( SLURM_ARRAY_TASK_ID / COMBOS_PER_RES ))
COMBO_INDEX=$(( SLURM_ARRAY_TASK_ID % COMBOS_PER_RES ))
RESOLUTION=${RESOLUTIONS[RESOLUTION_INDEX]}

# Resolution-dependent mesh for fair comparison across coarse grids.
# Mapping:
#   6->3, 9->2, 12->2, 15->1, 18->1, 30->0
if [[ -n "${MESH_SIZE:-}" ]]; then
  MESH_SIZE="${MESH_SIZE}"
  echo "Using user-provided MESH_SIZE=${MESH_SIZE} override"
else
  case "${RESOLUTION}" in
    6)
      MESH_SIZE=3
      ;;
    9|12)
      MESH_SIZE=2
      ;;
    15|18)
      MESH_SIZE=1
      ;;
    30)
      MESH_SIZE=0
      ;;
    *)
      echo "Unsupported RESOLUTION=${RESOLUTION}"
      exit 1
      ;;
  esac
fi

case "${COMBO_INDEX}" in
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
# Run name includes resolution + mesh tag.
RES_TAG="res${RESOLUTION%.0}"
RUN_NAME=${RUN_NAME:-${RES_TAG}_m${MESH_SIZE}_w${WIDTH}_mp${MSG}_h6_bs${BATCH_SIZE}}
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
echo "resolution setup: RESOLUTION=${RESOLUTION} MESH_SIZE=${MESH_SIZE} (task_id=${SLURM_ARRAY_TASK_ID})"

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
#   sbatch --array=0-7 scripts/training/train_graphcast.slurm         # only resolution=6
#   sbatch --array=8-15 scripts/training/train_graphcast.slurm        # only resolution=9
#   sbatch --export=MAX_STEPS=2000 scripts/training/train_graphcast.slurm
#   sbatch --export=BATCH_SIZE=2,PRECISION=fp32 scripts/training/train_graphcast.slurm
#   sbatch --export=DATA_PATH=data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr scripts/training/train_graphcast.slurm
