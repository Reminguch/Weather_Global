#!/bin/bash
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs

VANILLA_SCRIPT="scripts/experiments/active/7y_small_vanilla_w256_mp2_continue_to_200k.slurm"
MAMBA_SCRIPT="scripts/experiments/active/7y_small_mamba_from_200k_staged.slurm"

declare -A VANILLA_MEM=(
  [res2]=80G
  [res3]=48G
  [res6]=32G
)
declare -A VANILLA_TIME=(
  [res2]=04:00:00
  [res3]=03:00:00
  [res6]=02:00:00
)
declare -A FROZEN_MEM=(
  [res2]=96G
  [res3]=64G
  [res6]=40G
)
declare -A FROZEN_TIME=(
  [res2]=12:00:00
  [res3]=10:00:00
  [res6]=08:00:00
)
declare -A RELEASE_MEM=(
  [res2]=96G
  [res3]=64G
  [res6]=40G
)
declare -A RELEASE_TIME=(
  [res2]=08:00:00
  [res3]=06:00:00
  [res6]=04:00:00
)

declare -A BASE_NAME=(
  [res2]=vanilla_gc_7y_res2_m4_w256_mp2_h6_bs8_accum1_stream200k
  [res3]=vanilla_gc_7y_res3_m4_w256_mp2_h6_bs8_accum1_stream200k
  [res6]=vanilla_gc_7y_res6_m3_w256_mp2_h6_bs8_accum1_stream200k
)

BASE_ROOT=${BASE_ROOT:-artifacts/checkpoints/7_years/small_experiments}
DRY_RUN=${DRY_RUN:-0}

submit_or_print() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN:' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    printf '999999\n'
  else
    "$@"
  fi
}

for res_key in res2 res3 res6; do
  run_dir="${BASE_ROOT}/${BASE_NAME[$res_key]}"
  if [[ ! -d "${run_dir}" ]]; then
    echo "Missing renamed 200k run directory for ${res_key}: ${run_dir}" >&2
    exit 1
  fi
  if [[ ! -f "${run_dir}/ckpt_step150000.npz" ]]; then
    echo "Missing exact 150k checkpoint for ${res_key}: ${run_dir}/ckpt_step150000.npz" >&2
    exit 1
  fi
done

for res_key in res2 res3 res6; do
  vanilla_job=$(
    submit_or_print sbatch --parsable \
      --job-name "gc7y-${res_key}-200k" \
      --mem "${VANILLA_MEM[$res_key]}" \
      --time "${VANILLA_TIME[$res_key]}" \
      --export "ALL,RES_KEY=${res_key}" \
      "${VANILLA_SCRIPT}"
  )
  echo "${res_key} vanilla continuation job: ${vanilla_job}"

  frozen_job=$(
    submit_or_print sbatch --parsable \
      --dependency "afterok:${vanilla_job}" \
      --job-name "mamba-${res_key}-frozen50k" \
      --mem "${FROZEN_MEM[$res_key]}" \
      --time "${FROZEN_TIME[$res_key]}" \
      --array "0-7" \
      --export "ALL,RES_KEY=${res_key},STAGE=frozen,RESUME_FROM_LATEST=1" \
      "${MAMBA_SCRIPT}"
  )
  echo "${res_key} frozen Mamba array job: ${frozen_job}"

  release_job=$(
    submit_or_print sbatch --parsable \
      --dependency "afterok:${frozen_job}" \
      --job-name "mamba-${res_key}-release20k" \
      --mem "${RELEASE_MEM[$res_key]}" \
      --time "${RELEASE_TIME[$res_key]}" \
      --array "0-7" \
      --export "ALL,RES_KEY=${res_key},STAGE=release,RESUME_FROM_LATEST=1" \
      "${MAMBA_SCRIPT}"
  )
  echo "${res_key} release Mamba array job: ${release_job}"
done
