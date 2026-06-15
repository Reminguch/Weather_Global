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
MODE=${MODE:-probe}  # probe, chain, or all
STAGE_SCRIPT=${STAGE_SCRIPT:-scripts/experiments/active/res1_official_gc_mamba_bptt24_lora_release_stage.slurm}

OUT_DIR=${OUT_DIR:-artifacts/checkpoints/res1_official/gc_mamba_bptt24_lora_r4_allbptt_gclr3e7_mlr10x}
BASE_ROOT=${BASE_ROOT:-artifacts/checkpoints/res1_official/gc_mamba_bptt24_release_curriculum_gclr3e7_mlr10x}
STAGE1_RUN=${STAGE1_RUN:-stage1_k4_mamba10k_lr1e-4}
PROBE_RUN=${PROBE_RUN:-fit_probe_k4_mamba_lora_r4_allbptt_2steps}
STAGE2_RUN=${STAGE2_RUN:-stage2_k4_mamba_lora_r4_4k_allbptt_mlr3e-6}
STAGE3_RUN=${STAGE3_RUN:-stage3_k8_mamba_lora_r4_10k_allbptt_mlr3e-6}
STAGE4_RUN=${STAGE4_RUN:-stage4_k12_mamba_lora_r4_10k_allbptt_mlr3e-6}

STAGE1_CKPT="${BASE_ROOT}/${STAGE1_RUN}/ckpt_best.npz"
PROBE_CKPT="${OUT_DIR}/${PROBE_RUN}/ckpt_best.npz"
STAGE2_CKPT="${OUT_DIR}/${STAGE2_RUN}/ckpt_best.npz"
STAGE3_CKPT="${OUT_DIR}/${STAGE3_RUN}/ckpt_best.npz"
STAGE4_CKPT="${OUT_DIR}/${STAGE4_RUN}/ckpt_best.npz"

LR=${LR:-3e-6}
MAMBA_LR=${MAMBA_LR:-3e-6}
LORA_LR=${LORA_LR:-3e-6}
LORA_RANK=${LORA_RANK:-4}
LORA_ALPHA=${LORA_ALPHA:-4}
LORA_SCOPE=${LORA_SCOPE:-processor_mlp}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
ADAMW_BETA2=${ADAMW_BETA2:-0.95}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-32}

PROBE_TIME_LIMIT=${PROBE_TIME_LIMIT:-04:00:00}
STAGE2_TIME_LIMIT=${STAGE2_TIME_LIMIT:-24:00:00}
STAGE3_TIME_LIMIT=${STAGE3_TIME_LIMIT:-48:00:00}
STAGE4_TIME_LIMIT=${STAGE4_TIME_LIMIT:-48:00:00}
MEMORY_GB=${MEMORY_GB:-320}

case "${MODE}" in
  probe|chain|all) ;;
  *)
    echo "MODE must be one of: probe, chain, all (got ${MODE})" >&2
    exit 1
    ;;
esac

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "Missing stage-1 checkpoint: ${STAGE1_CKPT}" >&2
  exit 1
fi

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

COMMON="ALL,OUT_DIR=${OUT_DIR},LEN_SEGMENT=96,BPTT_STEPS=24,CHUNK_LOAD_WORKERS=6,BATCH_SIZE=1,EVAL_BATCH_SIZE=1,DATA_CACHE_MODE=never,BATCH_BUILDER=prepared_array,PRECISION=bf16,TEMPORAL_INSERT_COUNT=1,TEMPORAL_LAYERS=2,D_INNER=128,D_STATE=16,MEMORY_MODE=optimal,AR_LOSS_MODE=all_bptt_uniform,LR=${LR},MAMBA_LR=${MAMBA_LR},LORA_LR=${LORA_LR},LORA_RANK=${LORA_RANK},LORA_ALPHA=${LORA_ALPHA},LORA_SCOPE=${LORA_SCOPE},WEIGHT_DECAY=${WEIGHT_DECAY},ADAMW_BETA2=${ADAMW_BETA2},MAX_GRAD_NORM=${MAX_GRAD_NORM}"
FULL_EVAL="EVAL_NUM_SEGMENTS=16,FINAL_EVAL_NUM_SEGMENTS=all,EVAL_EVERY=2000,CHECKPOINT_EVERY=2000,AR_GRADIENT_ALIGNMENT_DIAGNOSTICS=1,AR_GRADIENT_ALIGNMENT_EVERY=2000,AR_GRADIENT_ALIGNMENT_NUM_CHUNKS=1"

echo "GC-Mamba LoRA all-BPTT release curriculum"
echo "  mode: ${MODE}"
echo "  dry run: ${DRY_RUN}"
echo "  stage1 ckpt: ${STAGE1_CKPT}"
echo "  output root: ${OUT_DIR}"
echo "  final planned ckpt: ${STAGE4_CKPT}"

probe_job=""
if [[ "${MODE}" == "probe" || "${MODE}" == "all" ]]; then
  submit_stage \
    "probe" \
    "" \
    "res1-gcm-lora-probe-k4" \
    "${COMMON},RUN_NAME=${PROBE_RUN},TARGET_STEPS=4,MAX_STEPS=2,EVAL_NUM_SEGMENTS=1,FINAL_EVAL_NUM_SEGMENTS=1,EVAL_EVERY=999999,CHECKPOINT_EVERY=999999,AR_GRADIENT_ALIGNMENT_DIAGNOSTICS=0,CKPT_IN=${STAGE1_CKPT}" \
    "${PROBE_TIME_LIMIT}" \
    "${MEMORY_GB}"
  probe_job="${SUBMITTED_JOB_ID}"
fi

if [[ "${MODE}" == "probe" ]]; then
  echo "Probe-only mode complete."
  echo "After it succeeds, inspect ${OUT_DIR}/${PROBE_RUN}/{actual_usage.json,memory_gib.json} and submit MODE=chain."
  exit 0
fi

stage2_dependency=""
if [[ "${MODE}" == "all" ]]; then
  stage2_dependency="${probe_job}"
fi

submit_stage \
  "stage2" \
  "${stage2_dependency}" \
  "res1-gcm-lora-s2-k4" \
  "${COMMON},${FULL_EVAL},RUN_NAME=${STAGE2_RUN},TARGET_STEPS=4,MAX_STEPS=4000,CKPT_IN=${STAGE1_CKPT}" \
  "${STAGE2_TIME_LIMIT}" \
  "${MEMORY_GB}"
stage2_job="${SUBMITTED_JOB_ID}"

submit_stage \
  "stage3" \
  "${stage2_job}" \
  "res1-gcm-lora-s3-k8" \
  "${COMMON},${FULL_EVAL},RUN_NAME=${STAGE3_RUN},TARGET_STEPS=8,MAX_STEPS=10000,CKPT_IN=${STAGE2_CKPT}" \
  "${STAGE3_TIME_LIMIT}" \
  "${MEMORY_GB}"
stage3_job="${SUBMITTED_JOB_ID}"

submit_stage \
  "stage4" \
  "${stage3_job}" \
  "res1-gcm-lora-s4-k12" \
  "${COMMON},${FULL_EVAL},RUN_NAME=${STAGE4_RUN},TARGET_STEPS=12,MAX_STEPS=10000,CKPT_IN=${STAGE3_CKPT}" \
  "${STAGE4_TIME_LIMIT}" \
  "${MEMORY_GB}"
stage4_job="${SUBMITTED_JOB_ID}"

echo "Curriculum chain ready:"
if [[ -n "${probe_job}" ]]; then
  echo "  probe job: ${probe_job}"
fi
echo "  stage2 job: ${stage2_job}"
echo "  stage3 job: ${stage3_job}"
echo "  stage4 job: ${stage4_job}"
echo "Final planned best checkpoint: ${STAGE4_CKPT}"
