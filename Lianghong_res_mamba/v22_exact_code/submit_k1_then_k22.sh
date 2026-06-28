#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

STAGE_SCRIPT="${STAGE_SCRIPT:-Lianghong_res_mamba/v22_exact_code/run_v22_stage.slurm}"
OUT_DIR="${OUT_DIR:-artifacts/checkpoints/lianghong_v22_exact_code_iv9432}"
K1_RUN="${K1_RUN:-K1_from_gcsmall_iv9432_23k}"
K22_RUN="${K22_RUN:-K22_from_K1_iv9432_23k}"
RESIDUAL_ROOT="${RESIDUAL_ROOT:-data/graphcast/graphcast/dataset/precomputed_residuals/lianghong_v22_iv9432_res1}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"

K1_CKPT="${OUT_DIR}/${K1_RUN}/v13_residual_step23000.pkl"

bash Lianghong_res_mamba/v22_exact_code/preflight.sh

if [[ ! -d "${RESIDUAL_ROOT}" ]]; then
  echo "Missing precomputed residual root required by v22 trainer: ${RESIDUAL_ROOT}" >&2
  echo "Create this v22-format residual root before submitting training." >&2
  exit 2
fi

check_target() {
  local path="$1"
  if [[ -e "${path}" && "${ALLOW_EXISTING}" != "1" ]]; then
    echo "Refusing to submit because target exists: ${path}" >&2
    echo "Set ALLOW_EXISTING=1 to override." >&2
    exit 2
  fi
}

check_target "${OUT_DIR}/${K1_RUN}"
check_target "${OUT_DIR}/${K22_RUN}"

k1_cmd=(
  sbatch --parsable
  --time=48:00:00
  --mem=340G
  --job-name=lh-v22-k1-exact
  --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",OUT_DIR="${OUT_DIR}",RUN_NAME="${K1_RUN}",MAX_STEPS=23000,START_STEP=1,AR_TAIL_K=0,RESIDUAL_ROOT="${RESIDUAL_ROOT}"
  "${STAGE_SCRIPT}"
)

k22_cmd_prefix=(
  sbatch --parsable
  --time=24:00:00
  --mem=340G
  --job-name=lh-v22-k22-exact
)

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] K=1 command:"
  printf ' %q' "${k1_cmd[@]}"
  printf '\n'
  echo "[dry-run] K=22 command will add --dependency=afterok:<K1_JOB_ID>:"
  printf ' %q' "${k22_cmd_prefix[@]}"
  printf ' %q' --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",OUT_DIR="${OUT_DIR}",RUN_NAME="${K22_RUN}",MAX_STEPS=26000,START_STEP=23001,AR_TAIL_K=22,RESUME_FROM="${K1_CKPT}",RESIDUAL_ROOT="${RESIDUAL_ROOT}"
  printf ' %q\n' "${STAGE_SCRIPT}"
  exit 0
fi

k1_job="$("${k1_cmd[@]}")"
echo "Submitted K=1: ${k1_job}"

k22_job="$(
  "${k22_cmd_prefix[@]}" \
    --dependency="afterok:${k1_job}" \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",OUT_DIR="${OUT_DIR}",RUN_NAME="${K22_RUN}",MAX_STEPS=26000,START_STEP=23001,AR_TAIL_K=22,RESUME_FROM="${K1_CKPT}",RESIDUAL_ROOT="${RESIDUAL_ROOT}" \
    "${STAGE_SCRIPT}"
)"
echo "Submitted K=22: ${k22_job} (afterok:${k1_job})"
echo "Monitor:"
echo "  squeue -j ${k1_job},${k22_job}"

