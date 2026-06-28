#!/bin/bash
#SBATCH --job-name=res1-gcm-lora-r4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=logs/res1_gc_mamba_lora_release_%x_%j.out
#SBATCH --error=logs/res1_gc_mamba_lora_release_%x_%j.err

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
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

source scripts/graphcast_env.sh

DATA_PATH=${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}
DATA_SOURCE=${DATA_SOURCE:-prepared_array}
PREPARED_DATA_ROOT=${PREPARED_DATA_ROOT:-data/graphcast/graphcast/dataset/prepared_stream}
STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}

BASE_ROOT=${BASE_ROOT:-artifacts/checkpoints/res1_official/gc_mamba_bptt24_release_curriculum_gclr3e7_mlr10x}
CKPT_IN=${CKPT_IN:-${BASE_ROOT}/stage1_k4_mamba10k_lr1e-4/ckpt_best.npz}
OUT_DIR=${OUT_DIR:-artifacts/checkpoints/res1_official/gc_mamba_bptt24_lora_r4_allbptt_gclr3e7_mlr10x}
RUN_NAME=${RUN_NAME:-stage2_k4_mamba_lora_r4_4k_allbptt_mlr3e-6}

RESOLUTION=${RESOLUTION:-1.0}
MESH_SIZE=${MESH_SIZE:-5}
WIDTH=${WIDTH:-512}
MSG=${MSG:-16}
INPUT_DURATION=${INPUT_DURATION:-12h}
VAL_YEAR=${VAL_YEAR:-2022}
TRAIN_START_YEAR=${TRAIN_START_YEAR:-2015}
TRAIN_END_YEAR=${TRAIN_END_YEAR:-2021}

TARGET_STEPS=${TARGET_STEPS:-4}
LEN_SEGMENT=${LEN_SEGMENT:-96}
BPTT_STEPS=${BPTT_STEPS:-24}
CHUNK_LOAD_WORKERS=${CHUNK_LOAD_WORKERS:-6}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
EVAL_NUM_SEGMENTS=${EVAL_NUM_SEGMENTS:-16}
FINAL_EVAL_NUM_SEGMENTS=${FINAL_EVAL_NUM_SEGMENTS:-all}
MAX_STEPS=${MAX_STEPS:-4000}
EVAL_EVERY=${EVAL_EVERY:-2000}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-2000}
DATA_CACHE_MODE=${DATA_CACHE_MODE:-never}
BATCH_BUILDER=${BATCH_BUILDER:-prepared_array}

LR=${LR:-3e-6}
MAMBA_LR=${MAMBA_LR:-3e-6}
LORA_LR=${LORA_LR:-3e-6}
LORA_RANK=${LORA_RANK:-4}
LORA_ALPHA=${LORA_ALPHA:-4}
LORA_SCOPE=${LORA_SCOPE:-processor_mlp}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
ADAMW_BETA1=${ADAMW_BETA1:-0.9}
ADAMW_BETA2=${ADAMW_BETA2:-0.95}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-32}

PRECISION=${PRECISION:-bf16}
SEED=${SEED:-0}
TEMPORAL_INSERT_COUNT=${TEMPORAL_INSERT_COUNT:-1}
TEMPORAL_LAYERS=${TEMPORAL_LAYERS:-2}
D_INNER=${D_INNER:-128}
D_STATE=${D_STATE:-16}
MEMORY_MODE=${MEMORY_MODE:-optimal}
PROCESSOR_REMAT_GROUP_SIZE=${PROCESSOR_REMAT_GROUP_SIZE:-4}
AR_LOSS_MODE=${AR_LOSS_MODE:-all_bptt_uniform}
AR_GRADIENT_ALIGNMENT_DIAGNOSTICS=${AR_GRADIENT_ALIGNMENT_DIAGNOSTICS:-1}
AR_GRADIENT_ALIGNMENT_EVERY=${AR_GRADIENT_ALIGNMENT_EVERY:-2000}
AR_GRADIENT_ALIGNMENT_NUM_CHUNKS=${AR_GRADIENT_ALIGNMENT_NUM_CHUNKS:-1}
DRY_RUN=${DRY_RUN:-0}

if [[ ! -f "${CKPT_IN}" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY RUN warning: missing input checkpoint: ${CKPT_IN}" >&2
  else
    echo "Missing input checkpoint: ${CKPT_IN}" >&2
    exit 1
  fi
fi
if [[ ! -d "${PREPARED_DATA_ROOT}/res1" ]]; then
  echo "Missing prepared res1 data store: ${PREPARED_DATA_ROOT}/res1" >&2
  exit 1
fi

echo "GC-Mamba LoRA release: target_steps=${TARGET_STEPS} lora_rank=${LORA_RANK}"
echo "Run directory: ${OUT_DIR}/${RUN_NAME}"
echo "Checkpoint in: ${CKPT_IN}"
echo "LRs: lr=${LR} mamba_lr=${MAMBA_LR} lora_lr=${LORA_LR}"
echo "AdamW: weight_decay=${WEIGHT_DECAY} beta1=${ADAMW_BETA1} beta2=${ADAMW_BETA2} max_grad_norm=${MAX_GRAD_NORM}"
echo "Memory: mode=${MEMORY_MODE} processor_remat_group_size=${PROCESSOR_REMAT_GROUP_SIZE:-none}"
echo "AR loss: ${AR_LOSS_MODE}; gradient diagnostics=${AR_GRADIENT_ALIGNMENT_DIAGNOSTICS} every=${AR_GRADIENT_ALIGNMENT_EVERY} chunks=${AR_GRADIENT_ALIGNMENT_NUM_CHUNKS}"

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
  --batch-builder "${BATCH_BUILDER}"
  --max-steps "${MAX_STEPS}"
  --eval-every "${EVAL_EVERY}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --lr "${LR}"
  --mamba-lr "${MAMBA_LR}"
  --lora-lr "${LORA_LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --adamw-beta1 "${ADAMW_BETA1}"
  --adamw-beta2 "${ADAMW_BETA2}"
  --max-grad-norm "${MAX_GRAD_NORM}"
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
  --ckpt-in "${CKPT_IN}"
  --resume-step 0
  --trainable-part mamba_lora
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-scope "${LORA_SCOPE}"
  --zero-init-temporal-out
)

if [[ -n "${PROCESSOR_REMAT_GROUP_SIZE}" ]]; then
  CMD+=(--processor-remat-group-size "${PROCESSOR_REMAT_GROUP_SIZE}")
fi

if [[ "${AR_GRADIENT_ALIGNMENT_DIAGNOSTICS}" == "1" ]]; then
  CMD+=(
    --ar-gradient-alignment-diagnostics
    --ar-gradient-alignment-every "${AR_GRADIENT_ALIGNMENT_EVERY}"
    --ar-gradient-alignment-num-chunks "${AR_GRADIENT_ALIGNMENT_NUM_CHUNKS}"
  )
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Resolved command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
