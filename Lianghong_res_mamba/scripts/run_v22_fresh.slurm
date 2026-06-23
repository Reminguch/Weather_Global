#!/bin/bash
#SBATCH --job-name=lianghong-v22-fresh
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=logs/lianghong_v22_fresh_%j.out
#SBATCH --error=logs/lianghong_v22_fresh_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/graphcast_env.sh" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
if [[ ! -f "${PROJECT_ROOT}/scripts/graphcast_env.sh" ]]; then
  echo "Invalid PROJECT_ROOT: ${PROJECT_ROOT}" >&2
  echo "Set PROJECT_ROOT to the Weather_global repo root before submitting." >&2
  exit 2
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
mkdir -p .tmp/matplotlib
export MPLCONFIGDIR="${PROJECT_ROOT}/.tmp/matplotlib"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

source scripts/graphcast_env.sh

DATA_PATH=${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}
DATA_SOURCE=${DATA_SOURCE:-prepared_array}
PREPARED_DATA_ROOT=${PREPARED_DATA_ROOT:-data/graphcast/graphcast/dataset/prepared_stream}
STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}
OFFICIAL_CKPT=${OFFICIAL_CKPT:-"data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"}

OUT_DIR=${OUT_DIR:-artifacts/checkpoints/lianghong_res_mamba}
RUN_NAME=${RUN_NAME:-v22_fresh_residual_mamba_k22}

RESOLUTION=${RESOLUTION:-1.0}
MESH_SIZE=${MESH_SIZE:-5}
WIDTH=${WIDTH:-512}
MSG=${MSG:-2}
INPUT_DURATION=${INPUT_DURATION:-12h}
VAL_YEAR=${VAL_YEAR:-2022}
TRAIN_START_YEAR=${TRAIN_START_YEAR:-2015}
TRAIN_END_YEAR=${TRAIN_END_YEAR:-2021}

TARGET_STEPS=${TARGET_STEPS:-22}
LEN_SEGMENT=${LEN_SEGMENT:-96}
BPTT_STEPS=${BPTT_STEPS:-24}
MAX_STEPS=${MAX_STEPS:-26000}

CHUNK_LOAD_WORKERS=${CHUNK_LOAD_WORKERS:-6}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
EVAL_NUM_SEGMENTS=${EVAL_NUM_SEGMENTS:-16}
FINAL_EVAL_NUM_SEGMENTS=${FINAL_EVAL_NUM_SEGMENTS:-all}
EVAL_EVERY=${EVAL_EVERY:-2000}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-500}
DATA_CACHE_MODE=${DATA_CACHE_MODE:-never}
DATA_CACHE_MAX_GIB=${DATA_CACHE_MAX_GIB:-48}
BATCH_BUILDER=${BATCH_BUILDER:-prepared_array}

LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
PRECISION=${PRECISION:-bf16}
SEED=${SEED:-22}
MEMORY_MODE=${MEMORY_MODE:-optimal}
AR_LOSS_MODE=${AR_LOSS_MODE:-tail_uniform}
RESIDUAL_AR_FEEDBACK=${RESIDUAL_AR_FEEDBACK:-baseline_plus_residual}
RESIDUAL_OUTPUT_HEAD=${RESIDUAL_OUTPUT_HEAD:-enabled}

TEMPORAL_INSERT_COUNT=${TEMPORAL_INSERT_COUNT:-2}
TEMPORAL_LAYERS=${TEMPORAL_LAYERS:-2}
D_INNER=${D_INNER:-128}
D_STATE=${D_STATE:-16}
D_CONV=${D_CONV:-4}
DT_RANK=${DT_RANK:-auto}
DRY_RUN=${DRY_RUN:-0}

if [[ ! -f "${OFFICIAL_CKPT}" ]]; then
  echo "Missing official checkpoint: ${OFFICIAL_CKPT}" >&2
  exit 2
fi
if [[ ! -d "${PREPARED_DATA_ROOT}/res1" ]]; then
  echo "Missing prepared res1 data store: ${PREPARED_DATA_ROOT}/res1" >&2
  exit 2
fi

RUN_DIR="${OUT_DIR}/${RUN_NAME}"
echo "Lianghong v22 fresh residual-Mamba run"
echo "Run directory: ${RUN_DIR}"
echo "target_steps=${TARGET_STEPS}, bptt_steps=${BPTT_STEPS}, len_segment=${LEN_SEGMENT}"

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
  --residual-output-head "${RESIDUAL_OUTPUT_HEAD}"
  --temporal-backbone mamba
  --temporal-location mesh_processor_interleaved
  --temporal-stateful
  --temporal-d-inner "${D_INNER}"
  --temporal-d-state "${D_STATE}"
  --temporal-d-conv "${D_CONV}"
  --temporal-dt-rank "${DT_RANK}"
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
