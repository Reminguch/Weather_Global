#!/bin/bash
#SBATCH --job-name=gc-finetune-full
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gc_finetune_%j.out
#SBATCH --error=logs/gc_finetune_%j.err
#
# How finetuning works:
#   - Each sample uses 2 input steps (from task config) + target_steps forecast steps (6h each).
#   - Loss = one-step loss (first 6h) + rollout_weight * rollout loss over all target steps.
#   - target_steps is set by ROLLOUT_STEPS. So ROLLOUT_STEPS=4  → 24h forecast, ROLLOUT_STEPS=24 → 6-day forecast.
#   - To optimize 6-day forecast set FORECAST_DAYS=6 (or ROLLOUT_STEPS=24).
#
set -euo pipefail

# Resolve project root from submit dir (preferred) or script location.
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

WB2_STREAM_URI=${WB2_STREAM_URI:-gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr}
TRAIN_PATH=${TRAIN_PATH:-${WB2_STREAM_URI}}
EVAL_PATH=${EVAL_PATH:-${WB2_STREAM_URI}}
CKPT_IN=${CKPT_IN:-"data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"}
OUT_DIR=${OUT_DIR:-artifacts/checkpoints/graphcast_finetune_1y}
RUN_NAME=${RUN_NAME:-a100_full}

BATCH_SIZE=${BATCH_SIZE:-1}
EPOCHS=${EPOCHS:-5}
EVAL_EVERY=${EVAL_EVERY:-100}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-2}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-200}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
SEED=${SEED:-0}
# Forecast length: 1 = 24h (4 steps), 6 = 6-day (24 steps). Sets rollout target steps.
FORECAST_DAYS=${FORECAST_DAYS:-6}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-$((${FORECAST_DAYS} * 4))}
ROLLOUT_WEIGHT=${ROLLOUT_WEIGHT:-0.5}
PRECISION=${PRECISION:-bf16}
EXTRA_ARGS=${EXTRA_ARGS:-}

EXTRA_ARGS_ARR=()
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARR=(${EXTRA_ARGS})
fi

python -u scripts/training/finetune_graphcast.py \
  --train-path "${TRAIN_PATH}" \
  --eval-path "${EVAL_PATH}" \
  --ckpt-in "${CKPT_IN}" \
  --out-dir "${OUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --eval-every "${EVAL_EVERY}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --seed "${SEED}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --rollout-weight "${ROLLOUT_WEIGHT}" \
  --precision "${PRECISION}" \
  "${EXTRA_ARGS_ARR[@]}"

# Example submissions:
#   sbatch scripts/training/finetune_graphcast.slurm
#   sbatch --export=BATCH_SIZE=2,RUN_NAME=a100_bs2 scripts/training/finetune_graphcast.slurm
#   sbatch --export=EXTRA_ARGS="--no-rollout",RUN_NAME=a100_full_noroll scripts/training/finetune_graphcast.slurm
#   sbatch --export=EPOCHS=0,RUN_NAME=full_smoke_init scripts/training/finetune_graphcast.slurm
#   sbatch --export=EVAL_BATCH_SIZE=4,RUN_NAME=a100_eval4 scripts/training/finetune_graphcast.slurm
#   6-day forecast (24 rollout steps; may need BATCH_SIZE=1 and more memory/time):
#   sbatch --export=FORECAST_DAYS=6,RUN_NAME=a100_6d scripts/training/finetune_graphcast.slurm
