#!/bin/bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

DRY_RUN=${DRY_RUN:-1}
OUT_DIR=${OUT_DIR:-artifacts/checkpoints/res1_official/gc_mamba_bptt24_lora_r4_gclr3e7_mlr10x}
RUN_NAME=${RUN_NAME:-stage2_k4_mamba_lora_r4_4k_mlr3e-6}
CKPT_PATH=${CKPT_PATH:-${OUT_DIR}/${RUN_NAME}/ckpt_best.npz}

EXPORTS=(
  "FAMILIES=gc_mamba"
  "RESOLUTIONS=1"
  "CHECKPOINT_PATHS=${CKPT_PATH}"
  "CHECKPOINT_ROOTS="
  "EVAL_MODES=${EVAL_MODES:-cold warm}"
  "METRICS=${METRICS:-weighted_allvars per_variable}"
  "WARMUP_STEPS=${WARMUP_STEPS:-24}"
  "TRUNK_STEPS=${TRUNK_STEPS:-32}"
  "RES1_MEM=${RES1_MEM:-30G}"
  "DEFAULT_ARRAY_TIME=${DEFAULT_ARRAY_TIME:-00:40:00}"
  "OUTPUT_CSV_SUFFIX=${OUTPUT_CSV_SUFFIX:-_lora_r4}"
  "SHARD_DATA_DIR=${SHARD_DATA_DIR:-plots/analyze_models/data/resolution_eval/res1_official_gc_mamba_lora_r4/shards}"
)

echo "GC-Mamba LoRA resolution eval"
echo "  checkpoint: ${CKPT_PATH}"
echo "  DRY_RUN=${DRY_RUN}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY RUN env:'
  printf ' %q' "${EXPORTS[@]}"
  printf ' bash scripts/analyze_models/submit_resolution_eval_array.sh\n'
  exit 0
fi

for item in "${EXPORTS[@]}"; do
  export "${item}"
done

bash scripts/analyze_models/submit_resolution_eval_array.sh
