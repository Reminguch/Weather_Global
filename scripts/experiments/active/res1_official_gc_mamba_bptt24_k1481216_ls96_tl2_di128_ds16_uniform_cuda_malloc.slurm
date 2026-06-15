#!/bin/bash
#SBATCH --job-name=res1-gcm-b24-ls96-u
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/res1_official_gc_mamba_bptt24_ls96_uniform_cuda_%x_%A_%a.out
#SBATCH --error=logs/res1_official_gc_mamba_bptt24_ls96_uniform_cuda_%x_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
mkdir -p .tmp/matplotlib
export MPLCONFIGDIR="${PROJECT_ROOT}/.tmp/matplotlib"
export TF_GPU_ALLOCATOR=cuda_malloc_async

source scripts/graphcast_env.sh

TARGET_STEPS_SET=(1 4 8 12 16)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
EXPECTED_TASKS=${#TARGET_STEPS_SET[@]}
if (( TASK_ID < 0 || TASK_ID >= EXPECTED_TASKS )); then
  echo "Task ${TASK_ID} outside expected range 0-$((EXPECTED_TASKS - 1))" >&2
  exit 1
fi

TARGET_STEPS="${TARGET_STEPS_SET[$TASK_ID]}"

DATA_PATH=${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}
DATA_SOURCE=${DATA_SOURCE:-prepared_array}
PREPARED_DATA_ROOT=${PREPARED_DATA_ROOT:-data/graphcast/graphcast/dataset/prepared_stream}
STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}
OFFICIAL_CKPT=${OFFICIAL_CKPT:-"data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"}

if [[ ! -f "${OFFICIAL_CKPT}" ]]; then
  echo "Missing official checkpoint: ${OFFICIAL_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${PREPARED_DATA_ROOT}/res1" ]]; then
  echo "Missing prepared res1 data store: ${PREPARED_DATA_ROOT}/res1" >&2
  exit 1
fi

OUT_DIR=${OUT_DIR:-artifacts/checkpoints/res1_official/gc_mamba_bptt24_k1481216_ls96_tl2_di128_ds16_optimal_allbptt_cuda_malloc}
RESOLUTION=${RESOLUTION:-1.0}
MESH_SIZE=${MESH_SIZE:-5}
WIDTH=${WIDTH:-512}
MSG=${MSG:-16}
INPUT_DURATION=${INPUT_DURATION:-12h}
VAL_YEAR=${VAL_YEAR:-2022}
TRAIN_START_YEAR=${TRAIN_START_YEAR:-2015}
TRAIN_END_YEAR=${TRAIN_END_YEAR:-2021}

MAX_STEPS=${MAX_STEPS:-20000}
LEN_SEGMENT=${LEN_SEGMENT:-96}
BPTT_STEPS=${BPTT_STEPS:-24}
CHUNK_LOAD_WORKERS=${CHUNK_LOAD_WORKERS:-6}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
EVAL_NUM_SEGMENTS=${EVAL_NUM_SEGMENTS:-16}
FINAL_EVAL_NUM_SEGMENTS=${FINAL_EVAL_NUM_SEGMENTS:-all}
EVAL_EVERY=${EVAL_EVERY:-2000}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-2000}
DATA_CACHE_MODE=${DATA_CACHE_MODE:-never}
DATA_CACHE_MAX_GIB=${DATA_CACHE_MAX_GIB:-48}
BATCH_BUILDER=${BATCH_BUILDER:-prepared_array}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
PRECISION=${PRECISION:-bf16}
SEED=${SEED:-0}
TEMPORAL_INSERT_COUNT=${TEMPORAL_INSERT_COUNT:-1}
TEMPORAL_LAYERS=${TEMPORAL_LAYERS:-2}
D_INNER=${D_INNER:-128}
D_STATE=${D_STATE:-16}
MEMORY_MODE=${MEMORY_MODE:-optimal}
AR_LOSS_MODE=${AR_LOSS_MODE:-all_bptt_uniform}
DRY_RUN=${DRY_RUN:-0}

RUN_NAME_SUFFIX=${RUN_NAME_SUFFIX:-}
RUN_NAME="official_res1_gcsmall_gc_mamba_tc${TEMPORAL_INSERT_COUNT}_tl${TEMPORAL_LAYERS}_di${D_INNER}_ds${D_STATE}_k${TARGET_STEPS}_bptt${BPTT_STEPS}_ls${LEN_SEGMENT}_bs${BATCH_SIZE}_${MEMORY_MODE}_allbptt_cuda_malloc${RUN_NAME_SUFFIX}"
RUN_DIR="${OUT_DIR}/${RUN_NAME}"

echo "Task ${TASK_ID}/${EXPECTED_TASKS}: model=gc_mamba layers=${TEMPORAL_LAYERS} d_inner=${D_INNER} d_state=${D_STATE} target_steps=${TARGET_STEPS}"
echo "Run directory: ${RUN_DIR}"
echo "TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR}"

CMD=(
  python -u -m src.models.mamba.training.segments_training
  --model gc_mamba
  --data-path "${DATA_PATH}"
  --data-source "${DATA_SOURCE}"
  --prepared-data-root "${PREPARED_DATA_ROOT}"
  --stats-dir "${STATS_DIR}"
  --out-dir "${OUT_DIR}"
  --run-name "${RUN_NAME}"
  --val-year "${VAL_YEAR}"
  --train-start-year "${TRAIN_START_YEAR}"
  --train-end-year "${TRAIN_END_YEAR}"
  --resolution "${RESOLUTION}"
  --mesh-size "${MESH_SIZE}"
  --width "${WIDTH}"
  --processor-msg-steps "${MSG}"
  --input-duration "${INPUT_DURATION}"
  --target-steps "${TARGET_STEPS}"
  --len-segment "${LEN_SEGMENT}"
  --bptt-steps "${BPTT_STEPS}"
  --autoregressive-loss-mode "${AR_LOSS_MODE}"
  --chunk-load-workers "${CHUNK_LOAD_WORKERS}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --eval-num-segments "${EVAL_NUM_SEGMENTS}"
  --final-eval-num-segments "${FINAL_EVAL_NUM_SEGMENTS}"
  --data-cache-mode "${DATA_CACHE_MODE}"
  --data-cache-max-gib "${DATA_CACHE_MAX_GIB}"
  --batch-builder "${BATCH_BUILDER}"
  --max-steps "${MAX_STEPS}"
  --eval-every "${EVAL_EVERY}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --precision "${PRECISION}"
  --seed "${SEED}"
  --memory-mode "${MEMORY_MODE}"
  --temporal-backbone mamba
  --temporal-location mesh_processor_interleaved
  --temporal-stateful
  --temporal-d-inner "${D_INNER}"
  --temporal-d-state "${D_STATE}"
  --temporal-d-conv 4
  --temporal-dt-rank auto
  --temporal-layers "${TEMPORAL_LAYERS}"
  --temporal-dropout 0.0
  --temporal-insert-count "${TEMPORAL_INSERT_COUNT}"
  --ckpt-in "${OFFICIAL_CKPT}"
  --init-from-graphcast-ckpt "${OFFICIAL_CKPT}"
  --trainable-part mamba
  --zero-init-temporal-out
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Resolved command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
