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
STAGE_SCRIPT=${STAGE_SCRIPT:-scripts/experiments/active/res1_official_gc_mamba_bptt24_lora_release_stage.slurm}
TIME_LIMIT=${TIME_LIMIT:-24:00:00}
MEMORY_GB=${MEMORY_GB:-320}

OUT_DIR=${OUT_DIR:-artifacts/checkpoints/res1_official/gc_mamba_bptt24_lora_r4_allbptt_gclr3e7_mlr10x}
RUN_NAME=${RUN_NAME:-stage2_k4_mamba_lora_r4_4k_allbptt_mlr3e-6}
BASE_ROOT=${BASE_ROOT:-artifacts/checkpoints/res1_official/gc_mamba_bptt24_release_curriculum_gclr3e7_mlr10x}
CKPT_IN=${CKPT_IN:-${BASE_ROOT}/stage1_k4_mamba10k_lr1e-4/ckpt_best.npz}

EXPORTS=(
  ALL
  "OUT_DIR=${OUT_DIR}"
  "RUN_NAME=${RUN_NAME}"
  "CKPT_IN=${CKPT_IN}"
  "LORA_RANK=${LORA_RANK:-4}"
  "LORA_ALPHA=${LORA_ALPHA:-4}"
  "LORA_LR=${LORA_LR:-3e-6}"
  "MAMBA_LR=${MAMBA_LR:-3e-6}"
  "MAX_STEPS=${MAX_STEPS:-4000}"
)
EXPORT_ARG=$(IFS=,; echo "${EXPORTS[*]}")

CMD=(
  sbatch
  --parsable
  --job-name=res1-gcm-lora-r4
  --time="${TIME_LIMIT}"
  --mem="${MEMORY_GB}G"
  --export="${EXPORT_ARG}"
  "${STAGE_SCRIPT}"
)

echo "GC-Mamba LoRA release"
echo "  checkpoint in: ${CKPT_IN}"
echo "  output ckpt: ${OUT_DIR}/${RUN_NAME}/ckpt_best.npz"
echo "  DRY_RUN=${DRY_RUN}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY RUN:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

job_raw=$("${CMD[@]}")
job_id="${job_raw%%;*}"
echo "Submitted LoRA release: ${job_id}"
echo "Monitor with: squeue -j ${job_id}"
