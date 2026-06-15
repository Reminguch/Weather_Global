#!/bin/bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs

DRY_RUN=${DRY_RUN:-1}
STAGE_SCRIPT=${STAGE_SCRIPT:-scripts/experiments/active/res1_official_gc_mamba_bptt24_release_curriculum_stage.slurm}
OUT_DIR=${OUT_DIR:-artifacts/checkpoints/res1_official/gc_mamba_bptt24_release_curriculum_gclr3e7_mlr10x}

GC_LR=${GC_LR:-3e-7}
MAMBA_RELEASE_LR=${MAMBA_RELEASE_LR:-3e-6}
MAMBA_STAGE1_LR=${MAMBA_STAGE1_LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
RELEASE_ADAMW_BETA2=${RELEASE_ADAMW_BETA2:-0.95}
RELEASE_MAX_GRAD_NORM=${RELEASE_MAX_GRAD_NORM:-32}

STAGE1_RUN=${STAGE1_RUN:-stage1_k4_mamba10k_lr1e-4}
STAGE2_RUN=${STAGE2_RUN:-stage2_k4_release4k_gclr3e-7_mlr3e-6}
STAGE3_RUN=${STAGE3_RUN:-stage3_k8_release10k_gclr3e-7_mlr3e-6}
STAGE4_RUN=${STAGE4_RUN:-stage4_k12_release10k_gclr3e-7_mlr3e-6}

STAGE1_CKPT="${OUT_DIR}/${STAGE1_RUN}/ckpt_best.npz"
STAGE2_CKPT="${OUT_DIR}/${STAGE2_RUN}/ckpt_best.npz"
STAGE3_CKPT="${OUT_DIR}/${STAGE3_RUN}/ckpt_best.npz"
STAGE4_CKPT="${OUT_DIR}/${STAGE4_RUN}/ckpt_best.npz"

echo "GC-Mamba staged release curriculum"
echo "OUT_DIR=${OUT_DIR}"
echo "DRY_RUN=${DRY_RUN}"
echo "Stage checkpoints:"
echo "  stage1 -> ${STAGE1_CKPT}"
echo "  stage2 -> ${STAGE2_CKPT}"
echo "  stage3 -> ${STAGE3_CKPT}"
echo "  stage4 -> ${STAGE4_CKPT}"

SUBMITTED_JOB_ID=""

submit_stage() {
  local placeholder="$1"
  local dependency="$2"
  local job_name="$3"
  local export_arg="$4"
  local time_limit="$5"
  local memory_gb="$6"

  local cmd=(
    sbatch
    --parsable
    --job-name="${job_name}"
    --time="${time_limit}"
    --mem="${memory_gb}G"
  )
  if [[ -n "${dependency}" ]]; then
    cmd+=(--dependency="afterok:${dependency}")
  fi
  cmd+=(--export="${export_arg}" "${STAGE_SCRIPT}")

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY RUN %s:' "${placeholder}"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    SUBMITTED_JOB_ID="${placeholder}_job_id"
    return
  fi

  local job_raw
  job_raw=$("${cmd[@]}")
  SUBMITTED_JOB_ID="${job_raw%%;*}"
  if [[ ! "${SUBMITTED_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "sbatch did not return a valid job id for ${placeholder}: ${job_raw}" >&2
    exit 1
  fi
  echo "Submitted ${placeholder}: ${SUBMITTED_JOB_ID}"
}

COMMON="ALL,OUT_DIR=${OUT_DIR},LEN_SEGMENT=96,BPTT_STEPS=24,BATCH_SIZE=1,EVAL_BATCH_SIZE=1,EVAL_NUM_SEGMENTS=16,FINAL_EVAL_NUM_SEGMENTS=all,EVAL_EVERY=2000,CHECKPOINT_EVERY=2000,DATA_CACHE_MODE=never,BATCH_BUILDER=prepared_array,TEMPORAL_INSERT_COUNT=1,TEMPORAL_LAYERS=2,D_INNER=128,D_STATE=16,MEMORY_MODE=optimal,AR_LOSS_MODE=all_bptt_uniform,WEIGHT_DECAY=${WEIGHT_DECAY}"

submit_stage \
  "stage1" \
  "" \
  "res1-gcm-s1-k4-mamba" \
  "${COMMON},STAGE_NAME=stage1,RUN_NAME=${STAGE1_RUN},TARGET_STEPS=4,MAX_STEPS=10000,TRAINABLE_PART=mamba,LR=${MAMBA_STAGE1_LR},MAMBA_LR=${MAMBA_STAGE1_LR},INIT_FROM_GRAPHCAST_CKPT=official" \
  "48:00:00" \
  "320"
stage1_job="${SUBMITTED_JOB_ID}"

submit_stage \
  "stage2" \
  "${stage1_job}" \
  "res1-gcm-s2-k4-release" \
  "${COMMON},STAGE_NAME=stage2,RUN_NAME=${STAGE2_RUN},TARGET_STEPS=4,MAX_STEPS=4000,TRAINABLE_PART=all,LR=${GC_LR},GRAPHCAST_LR=${GC_LR},MAMBA_LR=${MAMBA_RELEASE_LR},ADAMW_BETA2=${RELEASE_ADAMW_BETA2},MAX_GRAD_NORM=${RELEASE_MAX_GRAD_NORM},CKPT_IN=${STAGE1_CKPT},RESUME_STEP=0" \
  "24:00:00" \
  "320"
stage2_job="${SUBMITTED_JOB_ID}"

submit_stage \
  "stage3" \
  "${stage2_job}" \
  "res1-gcm-s3-k8-release" \
  "${COMMON},STAGE_NAME=stage3,RUN_NAME=${STAGE3_RUN},TARGET_STEPS=8,MAX_STEPS=10000,TRAINABLE_PART=all,LR=${GC_LR},GRAPHCAST_LR=${GC_LR},MAMBA_LR=${MAMBA_RELEASE_LR},ADAMW_BETA2=${RELEASE_ADAMW_BETA2},MAX_GRAD_NORM=${RELEASE_MAX_GRAD_NORM},CKPT_IN=${STAGE2_CKPT},RESUME_STEP=0" \
  "48:00:00" \
  "320"
stage3_job="${SUBMITTED_JOB_ID}"

submit_stage \
  "stage4" \
  "${stage3_job}" \
  "res1-gcm-s4-k12-release" \
  "${COMMON},STAGE_NAME=stage4,RUN_NAME=${STAGE4_RUN},TARGET_STEPS=12,MAX_STEPS=10000,TRAINABLE_PART=all,LR=${GC_LR},GRAPHCAST_LR=${GC_LR},MAMBA_LR=${MAMBA_RELEASE_LR},ADAMW_BETA2=${RELEASE_ADAMW_BETA2},MAX_GRAD_NORM=${RELEASE_MAX_GRAD_NORM},CKPT_IN=${STAGE3_CKPT},RESUME_STEP=0" \
  "48:00:00" \
  "320"
stage4_job="${SUBMITTED_JOB_ID}"

echo "Curriculum chain ready:"
echo "  stage1 job: ${stage1_job}"
echo "  stage2 job: ${stage2_job}"
echo "  stage3 job: ${stage3_job}"
echo "  stage4 job: ${stage4_job}"
echo "Final planned best checkpoint: ${STAGE4_CKPT}"
