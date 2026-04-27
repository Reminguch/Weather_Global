#!/bin/bash
#SBATCH --job-name=gc-mamba-seg-res-continue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/gc_mamba_seg_residual_continue_%A_%a.out
#SBATCH --error=logs/gc_mamba_seg_residual_continue_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
source scripts/graphcast_env.sh

COMBOS=(
  "mamba|128|1|16"
  "mamba|128|1|32"
  "mamba|128|2|16"
  "mamba|128|2|32"
  "mamba|256|1|16"
  "mamba|256|1|32"
  "mamba|256|2|16"
  "mamba|256|2|32"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#COMBOS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0-$(( ${#COMBOS[@]} - 1 ))"
  exit 1
fi

IFS="|" read -r KIND WIDTH MSG D_INNER <<< "${COMBOS[${TASK_ID}]}"

BASELINE_CKPT=${BASELINE_CKPT:-"data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"}
DATA_PATH=${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}
STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}
OUT_DIR=${OUT_DIR:-artifacts/checkpoints/graphcast_mamba_interleaved_segments_residual}

VAL_YEAR=${VAL_YEAR:-2021}
TRAIN_START_YEAR=${TRAIN_START_YEAR:-}
TRAIN_END_YEAR=${TRAIN_END_YEAR:-}

RESOLUTION=${RESOLUTION:-2.0}
MESH_SIZE=${MESH_SIZE:-4}
INPUT_DURATION=${INPUT_DURATION:-12h}
TARGET_STEPS=${TARGET_STEPS:-1}
LEN_SEGMENT=${LEN_SEGMENT:-30}
BPTT_STEPS=${BPTT_STEPS:-6}

# Keep continuation defaults aligned with the initial launch script. Res1
# residual segment training is memory-heavy on 80 GB A100s.
BATCH_SIZE=${BATCH_SIZE:-2}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-2}
CONTINUE_STEPS=${CONTINUE_STEPS:-20000}
EVAL_EVERY=${EVAL_EVERY:-2000}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-2000}
CHUNK_LOAD_WORKERS=${CHUNK_LOAD_WORKERS:-6}
DATA_CACHE_MODE=${DATA_CACHE_MODE:-auto}
DATA_CACHE_MAX_GIB=${DATA_CACHE_MAX_GIB:-48}
BATCH_BUILDER=${BATCH_BUILDER:-numpy}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
PRECISION=${PRECISION:-bf16}
SEED=${SEED:-0}

RUN_NAME="residual_mamba_int_seg_res2_m4_w${WIDTH}_mp${MSG}_di${D_INNER}_ds1_dc1_ls${LEN_SEGMENT}_bptt${BPTT_STEPS}_t1"
RUN_DIR="${OUT_DIR}/${RUN_NAME}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Run directory does not exist: ${RUN_DIR}"
  exit 1
fi

LATEST_CKPT=$(find "${RUN_DIR}" -maxdepth 1 -name 'ckpt_step*.npz' | sort -V | tail -n 1)
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "No checkpoints found in ${RUN_DIR}"
  exit 1
fi

RESUME_STEP=$(basename "${LATEST_CKPT}" .npz)
RESUME_STEP=${RESUME_STEP#ckpt_step}
if ! [[ "${RESUME_STEP}" =~ ^[0-9]+$ ]]; then
  echo "Could not parse step from checkpoint: ${LATEST_CKPT}"
  exit 1
fi

MAX_STEPS=$(( RESUME_STEP + CONTINUE_STEPS ))

echo "host: $(hostname)"
python --version
which python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "combo: kind=${KIND} width=${WIDTH} processor_msg_steps=${MSG} d_inner=${D_INNER}"
echo "segments: len_segment=${LEN_SEGMENT} bptt_steps=${BPTT_STEPS}"
echo "run: ${RUN_DIR}"
echo "resume: ckpt=${LATEST_CKPT} resume_step=${RESUME_STEP} continue_steps=${CONTINUE_STEPS} max_steps=${MAX_STEPS}"

CMD=(
  python -u scripts/training/residual_memory/train_graphcast_residual_segments.py
  --data-path "${DATA_PATH}"
  --stats-dir "${STATS_DIR}"
  --baseline-ckpt "${BASELINE_CKPT}"
  --resume-ckpt "${LATEST_CKPT}"
  --resume-step "${RESUME_STEP}"
  --out-dir "${OUT_DIR}"
  --run-name "${RUN_NAME}"
  --val-year "${VAL_YEAR}"
  --resolution "${RESOLUTION}"
  --mesh-size "${MESH_SIZE}"
  --width "${WIDTH}"
  --processor-msg-steps "${MSG}"
  --input-duration "${INPUT_DURATION}"
  --target-steps "${TARGET_STEPS}"
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
  --training-target residual
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --precision "${PRECISION}"
  --seed "${SEED}"
  --temporal-backbone mamba
  --temporal-location mesh_processor_interleaved
  --temporal-stateful
  --temporal-layers 1
  --temporal-d-inner "${D_INNER}"
  --temporal-d-state 1
  --temporal-d-conv 1
  --temporal-dt-rank auto
  --temporal-dropout 0.0
)

if [[ -n "${TRAIN_START_YEAR}" && -n "${TRAIN_END_YEAR}" ]]; then
  CMD+=(--train-start-year "${TRAIN_START_YEAR}" --train-end-year "${TRAIN_END_YEAR}")
fi

"${CMD[@]}"
