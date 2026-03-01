#!/bin/bash
#SBATCH --job-name=gc-res2-sweep
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/gc_res2_stream_%A_%a.out
#SBATCH --error=logs/gc_res2_stream_%A_%a.err

set -euo pipefail

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
    WIDTH=128
    MSG=2
    ;;
  2)
    WIDTH=256
    MSG=1
    ;;
  3)
    WIDTH=256
    MSG=2
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

WB2_URI=${WB2_URI:-gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr}
TRAIN_START=${TRAIN_START:-1979-01-01}
TRAIN_END=${TRAIN_END:-2020-12-31}
VAL_START=${VAL_START:-2021-01-01}
VAL_END=${VAL_END:-2021-12-31}
RESOLUTION=${RESOLUTION:-2.0}
MESH_SIZE=${MESH_SIZE:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_STEPS=${MAX_STEPS:-10000}
EVAL_EVERY=${EVAL_EVERY:-2000}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-2000}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
SEED=${SEED:-0}
PRECISION=${PRECISION:-bf16}
OUT_DIR=${OUT_DIR:-artifacts/checkpoints/graphcast_res2_stream}
CKPT_IN=${CKPT_IN:-"data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"}
STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}
RUN_NAME=${RUN_NAME:-res2_m4_w${WIDTH}_mp${MSG}_h6_bs${BATCH_SIZE}}

echo "resolved combo: WIDTH=${WIDTH} MSG=${MSG} RUN_NAME=${RUN_NAME}"

echo "command: python -u scripts/training/train_graphcast_res2_stream.py ..."
python -u scripts/training/train_graphcast_res2_stream.py \
  --wb2-uri "${WB2_URI}" \
  --train-start "${TRAIN_START}" \
  --train-end "${TRAIN_END}" \
  --val-start "${VAL_START}" \
  --val-end "${VAL_END}" \
  --resolution "${RESOLUTION}" \
  --mesh-size "${MESH_SIZE}" \
  --width "${WIDTH}" \
  --processor-msg-steps "${MSG}" \
  --batch-size "${BATCH_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --eval-every "${EVAL_EVERY}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --seed "${SEED}" \
  --precision "${PRECISION}" \
  --out-dir "${OUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --ckpt-in "${CKPT_IN}" \
  --stats-dir "${STATS_DIR}"

# Examples:
#   sbatch scripts/training/train_graphcast_res2_stream.slurm
#   sbatch --export=MAX_STEPS=2000 scripts/training/train_graphcast_res2_stream.slurm
#   sbatch --export=BATCH_SIZE=2,PRECISION=fp32 scripts/training/train_graphcast_res2_stream.slurm
