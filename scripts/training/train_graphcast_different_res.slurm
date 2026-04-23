#!/bin/bash
#SBATCH --job-name=gc-res-sweep
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-35
#SBATCH --output=logs/gc_stream_%A_%a.out
#SBATCH --error=logs/gc_stream_%A_%a.err

set -euo pipefail

# Resolution/mesh sweep script:
# - Sweeps message passing steps 1, 2, 3, 4
# - Sweeps widths 128, 256, 512
# - Uses mesh/resolution pairs:
#     mesh=2 -> res=15
#     mesh=3 -> res=4
#     mesh=4 -> res=1
# - Memory is set to 128G for the whole array because res=1 runs need
#   substantially more host RAM to enable the cached numpy batch path.
# - Omits mesh=1
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
#   3 (mesh,res) pairs x 4 mp values x 3 widths = 36 jobs total.
#   pair-major ordering:
#     task_id = pair_index * 12 + mp_index * 3 + width_index
NUM_WIDTHS=3
NUM_MPS=4
NUM_PAIRS=3
COMBOS_PER_PAIR=$(( NUM_MPS * NUM_WIDTHS ))
MAX_TASK_ID=$(( NUM_PAIRS * COMBOS_PER_PAIR - 1 ))

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > MAX_TASK_ID )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; expected [0, ${MAX_TASK_ID}]"
  exit 1
fi

WIDTH_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_WIDTHS ))
MP_INDEX=$(( (SLURM_ARRAY_TASK_ID % COMBOS_PER_PAIR) / NUM_WIDTHS ))
PAIR_INDEX=$(( SLURM_ARRAY_TASK_ID / COMBOS_PER_PAIR ))

case "${WIDTH_INDEX}" in
  0)
    WIDTH=128
    ;;
  1)
    WIDTH=256
    ;;
  2)
    WIDTH=512
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

case "${MP_INDEX}" in
  0)
    MSG=1
    ;;
  1)
    MSG=2
    ;;
  2)
    MSG=3
    ;;
  3)
    MSG=4
    ;;
  *)
    echo "Invalid MP_INDEX=${MP_INDEX}"
    exit 1
    ;;
esac

case "${PAIR_INDEX}" in
  0)
    MESH_SIZE=2
    RESOLUTION=15
    ;;
  1)
    MESH_SIZE=3
    RESOLUTION=4
    ;;
  2)
    MESH_SIZE=4
    RESOLUTION=1
    ;;
  *)
    echo "Invalid PAIR_INDEX=${PAIR_INDEX}"
    exit 1
    ;;
esac

DATA_PATH=${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}
VAL_YEAR=${VAL_YEAR:-2022}
TRAIN_START_YEAR=${TRAIN_START_YEAR:-}
TRAIN_END_YEAR=${TRAIN_END_YEAR:-}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_STEPS=${MAX_STEPS:-50000}
EVAL_EVERY=${EVAL_EVERY:-2000}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-2000}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
SEED=${SEED:-0}
PRECISION=${PRECISION:-bf16}
OUT_DIR=${OUT_DIR:-artifacts/checkpoints/res_mesh234_mp_sweep}
# Run name includes resolution + mesh tag.
RES_TAG="res${RESOLUTION%.0}"
RUN_NAME=${RUN_NAME:-${RES_TAG}_m${MESH_SIZE}_w${WIDTH}_mp${MSG}_h6_bs${BATCH_SIZE}}
CKPT_IN=${CKPT_IN:-"data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"}
STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}
AUTO_RESUME=${AUTO_RESUME:-0}
DATA_CACHE_MODE=${DATA_CACHE_MODE:-auto}
DATA_CACHE_MAX_GIB=${DATA_CACHE_MAX_GIB:-48}
BATCH_BUILDER=${BATCH_BUILDER:-numpy}
PREFETCH_WORKERS=${PREFETCH_WORKERS:-7}
PREFETCH_DEPTH=${PREFETCH_DEPTH:-8}
PREFETCH_DEVICE_DEPTH=${PREFETCH_DEVICE_DEPTH:-1}
USAGE_EVERY=${USAGE_EVERY:-50}

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
echo "resolution setup: RESOLUTION=${RESOLUTION} MESH_SIZE=${MESH_SIZE} PAIR_INDEX=${PAIR_INDEX} MP_INDEX=${MP_INDEX} (task_id=${SLURM_ARRAY_TASK_ID})"

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
  --data-cache-mode "${DATA_CACHE_MODE}"
  --data-cache-max-gib "${DATA_CACHE_MAX_GIB}"
  --batch-builder "${BATCH_BUILDER}"
  --prefetch-workers "${PREFETCH_WORKERS}"
  --prefetch-depth "${PREFETCH_DEPTH}"
  --prefetch-device-depth "${PREFETCH_DEVICE_DEPTH}"
  --usage-every "${USAGE_EVERY}"
)

if [[ -n "${RESUME_STEP:-}" ]]; then
  CMD+=(--resume-step "${RESUME_STEP}")
fi

if [[ -n "${TRAIN_START_YEAR}" && -n "${TRAIN_END_YEAR}" ]]; then
  CMD+=(--train-start-year "${TRAIN_START_YEAR}" --train-end-year "${TRAIN_END_YEAR}")
fi

"${CMD[@]}"

# Examples:
#   sbatch scripts/training/train_graphcast_different_res.slurm
#   sbatch --array=0-11 scripts/training/train_graphcast_different_res.slurm   # mesh=2, res=15, mp=1..4, widths 128/256/512
#   sbatch --array=12-23 scripts/training/train_graphcast_different_res.slurm  # mesh=3, res=4, mp=1..4, widths 128/256/512
#   sbatch --array=24-35 scripts/training/train_graphcast_different_res.slurm  # mesh=4, res=1, mp=1..4, widths 128/256/512
#   sbatch --array=0-2 scripts/training/train_graphcast_different_res.slurm    # mesh=2, res=15, mp=1, widths 128/256/512
#   sbatch --array=3-5 scripts/training/train_graphcast_different_res.slurm    # mesh=2, res=15, mp=2, widths 128/256/512
#   sbatch --array=6-8 scripts/training/train_graphcast_different_res.slurm    # mesh=2, res=15, mp=3, widths 128/256/512
#   sbatch --array=9-11 scripts/training/train_graphcast_different_res.slurm   # mesh=2, res=15, mp=4, widths 128/256/512
#   sbatch --export=MAX_STEPS=2000 scripts/training/train_graphcast_different_res.slurm
#   sbatch --export=BATCH_SIZE=2,PRECISION=fp32 scripts/training/train_graphcast_different_res.slurm
