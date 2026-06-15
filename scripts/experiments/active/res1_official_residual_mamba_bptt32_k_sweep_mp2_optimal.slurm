#!/bin/bash
#SBATCH --job-name=res1-resm-b32-mp2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/res1_official_residual_mamba_bptt32_mp2_optimal_%x_%A_%a.out
#SBATCH --error=logs/res1_official_residual_mamba_bptt32_mp2_optimal_%x_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
source scripts/graphcast_env.sh

TARGET_STEPS_SET=(8 16 24)

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

OUT_DIR=${OUT_DIR:-artifacts/checkpoints/res1_official/residual_mamba_bptt32_k_sweep_mp2_optimal}
RESOLUTION=${RESOLUTION:-1.0}
MESH_SIZE=${MESH_SIZE:-5}
WIDTH=${WIDTH:-512}
MSG=${MSG:-2}
INPUT_DURATION=${INPUT_DURATION:-12h}
VAL_YEAR=${VAL_YEAR:-2022}
TRAIN_START_YEAR=${TRAIN_START_YEAR:-2015}
TRAIN_END_YEAR=${TRAIN_END_YEAR:-2021}

MAX_STEPS=${MAX_STEPS:-20000}
LEN_SEGMENT=${LEN_SEGMENT:-64}
BPTT_STEPS=${BPTT_STEPS:-32}
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
TEMPORAL_INSERT_COUNT=${TEMPORAL_INSERT_COUNT:-2}
TEMPORAL_LAYERS=${TEMPORAL_LAYERS:-1}
D_INNER=${D_INNER:-128}
D_STATE=${D_STATE:-64}
RESIDUAL_AR_FEEDBACK=${RESIDUAL_AR_FEEDBACK:-baseline}
MEMORY_MODE=${MEMORY_MODE:-optimal}
DRY_RUN=${DRY_RUN:-0}

RUN_NAME_SUFFIX=${RUN_NAME_SUFFIX:-}
RUN_NAME="official_res1_gcsmall_residual_mamba_tc${TEMPORAL_INSERT_COUNT}_di${D_INNER}_ds${D_STATE}_k${TARGET_STEPS}_bptt${BPTT_STEPS}_ls${LEN_SEGMENT}_bs${BATCH_SIZE}_mp${MSG}_${MEMORY_MODE}${RUN_NAME_SUFFIX}"
RUN_DIR="${OUT_DIR}/${RUN_NAME}"

echo "Task ${TASK_ID}/${EXPECTED_TASKS}: model=residual_mamba d_inner=${D_INNER} d_state=${D_STATE} target_steps=${TARGET_STEPS}"
echo "Run directory: ${RUN_DIR}"
echo "Residual AR feedback: ${RESIDUAL_AR_FEEDBACK}"

CMD=(
  python -u -m src.models.mamba.training.segments_training
  --model residual_mamba
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
  --baseline-ckpt "${OFFICIAL_CKPT}"
  --residual-ar-feedback "${RESIDUAL_AR_FEEDBACK}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Resolved command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
