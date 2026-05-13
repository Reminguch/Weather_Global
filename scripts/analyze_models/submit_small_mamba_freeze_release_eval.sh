#!/bin/bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

source scripts/graphcast_env.sh

BASE_ROOT="artifacts/checkpoints/7_years/small_experiments"
DATA_BASE="plots/analyze_models/data/resolution_eval"
IMAGE_BASE="plots/analyze_models/images/resolution_eval"
PLOT_IMAGE_DIR="${IMAGE_BASE}/small_mamba_freeze_release_warm"

RESOLUTIONS="${RESOLUTIONS:-2 3 6}"
WARMUP_STEPS="${WARMUP_STEPS:-24}"
TRUNK_STEPS="${TRUNK_STEPS:-32}"
METRICS="${METRICS:-weighted_allvars per_variable}"
EVAL_MODES="${EVAL_MODES:-warm}"
RESIDUAL_EVAL_SEMANTICS="${RESIDUAL_EVAL_SEMANTICS:-teacher_forced_training_equivalent}"
LEAD_STEPS="${LEAD_STEPS:-}"
PLOT_LEAD_STEPS="${PLOT_LEAD_STEPS:-}"
OUTPUT_CSV_SUFFIX="${OUTPUT_CSV_SUFFIX:-}"
RES2_MEM="${RES2_MEM:-80G}"
RES2_TIME="${RES2_TIME:-04:00:00}"
OTHER_MEM="${OTHER_MEM:-48G}"
OTHER_TIME="${OTHER_TIME:-03:00:00}"
DRY_RUN="${DRY_RUN:-0}"

GC_SUITE="small_gc_mamba_freeze_release_warm"
RESIDUAL_SUITE="small_residual_mamba_freeze_release_warm"

VANILLA_ROOTS=(
  "${BASE_ROOT}/vanilla_gc_7y_res2_m4_w256_mp2_h6_bs8_accum1_stream200k"
  "${BASE_ROOT}/vanilla_gc_7y_res3_m4_w256_mp2_h6_bs8_accum1_stream200k"
  "${BASE_ROOT}/vanilla_gc_7y_res6_m3_w256_mp2_h6_bs8_accum1_stream200k"
)

GC_MAMBA_ROOTS=(
  "${BASE_ROOT}/small_mamba_frozen_from_vanilla_200k_50k/vanilla_gc_7y_res2_m4_w256_mp2_h6_bs8_accum1_stream200k_gc_mamba_tc2_di128_ds64_frozen50k"
  "${BASE_ROOT}/small_mamba_frozen_from_vanilla_200k_50k/vanilla_gc_7y_res3_m4_w256_mp2_h6_bs8_accum1_stream200k_gc_mamba_tc2_di128_ds64_frozen50k"
  "${BASE_ROOT}/small_mamba_frozen_from_vanilla_200k_50k/vanilla_gc_7y_res6_m3_w256_mp2_h6_bs8_accum1_stream200k_gc_mamba_tc2_di128_ds64_frozen50k"
  "${BASE_ROOT}/small_mamba_frozen_from_vanilla_200k_50k/vanilla_gc_7y_res2_m4_w256_mp2_h6_bs8_accum1_stream200k_gc_mamba_tc2_di256_ds128_frozen50k"
  "${BASE_ROOT}/small_mamba_frozen_from_vanilla_200k_50k/vanilla_gc_7y_res3_m4_w256_mp2_h6_bs8_accum1_stream200k_gc_mamba_tc2_di256_ds128_frozen50k"
  "${BASE_ROOT}/small_mamba_frozen_from_vanilla_200k_50k/vanilla_gc_7y_res6_m3_w256_mp2_h6_bs8_accum1_stream200k_gc_mamba_tc2_di256_ds128_frozen50k"
  "${BASE_ROOT}/small_mamba_release_from_frozen50k_20k/vanilla_gc_7y_res2_m4_w256_mp2_h6_bs8_accum1_stream200k_gc_mamba_tc2_di128_ds64_frozen50k_release20k"
  "${BASE_ROOT}/small_mamba_release_from_frozen50k_20k/vanilla_gc_7y_res6_m3_w256_mp2_h6_bs8_accum1_stream200k_gc_mamba_tc2_di128_ds64_frozen50k_release20k"
  "${BASE_ROOT}/small_mamba_release_from_frozen50k_20k/vanilla_gc_7y_res2_m4_w256_mp2_h6_bs8_accum1_stream200k_gc_mamba_tc2_di256_ds128_frozen50k_release20k"
  "${BASE_ROOT}/small_mamba_release_from_frozen50k_20k/vanilla_gc_7y_res3_m4_w256_mp2_h6_bs8_accum1_stream200k_gc_mamba_tc2_di256_ds128_frozen50k_release20k"
  "${BASE_ROOT}/small_mamba_release_from_frozen50k_20k/vanilla_gc_7y_res6_m3_w256_mp2_h6_bs8_accum1_stream200k_gc_mamba_tc2_di256_ds128_frozen50k_release20k"
)

RESIDUAL_MAMBA_ROOTS=(
  "${BASE_ROOT}/small_mamba_frozen_from_vanilla_200k_50k/vanilla_gc_7y_res2_m4_w256_mp2_h6_bs8_accum1_stream200k_residual_mamba_tc2_di128_ds64_frozen50k"
  "${BASE_ROOT}/small_mamba_frozen_from_vanilla_200k_50k/vanilla_gc_7y_res3_m4_w256_mp2_h6_bs8_accum1_stream200k_residual_mamba_tc2_di128_ds64_frozen50k"
  "${BASE_ROOT}/small_mamba_frozen_from_vanilla_200k_50k/vanilla_gc_7y_res6_m3_w256_mp2_h6_bs8_accum1_stream200k_residual_mamba_tc2_di128_ds64_frozen50k"
  "${BASE_ROOT}/small_mamba_frozen_from_vanilla_200k_50k/vanilla_gc_7y_res2_m4_w256_mp2_h6_bs8_accum1_stream200k_residual_mamba_tc2_di256_ds128_frozen50k"
  "${BASE_ROOT}/small_mamba_frozen_from_vanilla_200k_50k/vanilla_gc_7y_res3_m4_w256_mp2_h6_bs8_accum1_stream200k_residual_mamba_tc2_di256_ds128_frozen50k"
  "${BASE_ROOT}/small_mamba_frozen_from_vanilla_200k_50k/vanilla_gc_7y_res6_m3_w256_mp2_h6_bs8_accum1_stream200k_residual_mamba_tc2_di256_ds128_frozen50k"
  "${BASE_ROOT}/small_mamba_release_from_frozen50k_20k/vanilla_gc_7y_res2_m4_w256_mp2_h6_bs8_accum1_stream200k_residual_mamba_tc2_di128_ds64_frozen50k_release20k"
  "${BASE_ROOT}/small_mamba_release_from_frozen50k_20k/vanilla_gc_7y_res6_m3_w256_mp2_h6_bs8_accum1_stream200k_residual_mamba_tc2_di128_ds64_frozen50k_release20k"
  "${BASE_ROOT}/small_mamba_release_from_frozen50k_20k/vanilla_gc_7y_res2_m4_w256_mp2_h6_bs8_accum1_stream200k_residual_mamba_tc2_di256_ds128_frozen50k_release20k"
  "${BASE_ROOT}/small_mamba_release_from_frozen50k_20k/vanilla_gc_7y_res6_m3_w256_mp2_h6_bs8_accum1_stream200k_residual_mamba_tc2_di256_ds128_frozen50k_release20k"
)

validate_roots() {
  local root
  for root in "$@"; do
    if [[ ! -d "${root}" ]]; then
      echo "Missing checkpoint root: ${root}" >&2
      exit 1
    fi
    if [[ ! -f "${root}/ckpt_best.npz" ]]; then
      echo "Missing best checkpoint: ${root}/ckpt_best.npz" >&2
      exit 1
    fi
    if [[ ! -f "${root}/run_config.json" ]]; then
      echo "Missing run config: ${root}/run_config.json" >&2
      exit 1
    fi
  done
}

join_words() {
  local IFS=" "
  echo "$*"
}

submit_array_group() {
  local specs="$1"
  local mem="$2"
  local time_limit="$3"
  local checkpoint_roots="$4"
  local shard_data_dir="$5"
  local array_max
  local job_id
  read -r -a spec_array <<< "${specs}"
  array_max=$((${#spec_array[@]} - 1))
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY RUN: sbatch --parsable --mem=${mem} --time=${time_limit} --array=0-${array_max} --export=ALL,SHARD_SPECS='${specs}',SHARD_DATA_DIR='${shard_data_dir}',WARMUP_STEPS='${WARMUP_STEPS}',TRUNK_STEPS='${TRUNK_STEPS}',METRICS='${METRICS}',EVAL_MODES='${EVAL_MODES}',RESIDUAL_EVAL_SEMANTICS='${RESIDUAL_EVAL_SEMANTICS}',LEAD_STEPS='${LEAD_STEPS}',OUTPUT_CSV_SUFFIX='${OUTPUT_CSV_SUFFIX}',CHECKPOINT_ROOTS='<${#checkpoint_roots} chars>' scripts/analyze_models/run_resolution_eval_array.slurm" >&2
    echo "dry-array-${RANDOM}"
    return
  fi
  job_id=$(
    sbatch \
      --parsable \
      --mem="${mem}" \
      --time="${time_limit}" \
      --array="0-${array_max}" \
      --export=ALL,SHARD_SPECS="${specs}",SHARD_DATA_DIR="${shard_data_dir}",WARMUP_STEPS="${WARMUP_STEPS}",TRUNK_STEPS="${TRUNK_STEPS}",METRICS="${METRICS}",EVAL_MODES="${EVAL_MODES}",RESIDUAL_EVAL_SEMANTICS="${RESIDUAL_EVAL_SEMANTICS}",LEAD_STEPS="${LEAD_STEPS}",OUTPUT_CSV_SUFFIX="${OUTPUT_CSV_SUFFIX}",CHECKPOINT_ROOTS="${checkpoint_roots}" \
      scripts/analyze_models/run_resolution_eval_array.slurm
  )
  echo "${job_id}"
}

submit_suite() {
  local suite="$1"
  local families="$2"
  shift 2
  local roots=("$@")
  local checkpoint_roots shard_data_dir output_data_dir output_image_dir
  local discover_cmd shards shard_specs res2_specs other_specs job_id merge_dependency merge_job_id
  local dependency_job_ids=()

  validate_roots "${roots[@]}"

  checkpoint_roots="$(join_words "${roots[@]}")"
  shard_data_dir="${DATA_BASE}/${suite}/shards"
  output_data_dir="${DATA_BASE}/${suite}"
  output_image_dir="${IMAGE_BASE}/${suite}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY RUN: would create ${shard_data_dir}, ${output_data_dir}, ${output_image_dir}, logs"
  else
    mkdir -p "${shard_data_dir}" "${output_data_dir}" "${output_image_dir}" logs
  fi

  read -r -a discover_cmd <<< "python -u scripts/analyze_models/unified_resolution_eval.py"
  mapfile -t shards < <(
    "${discover_cmd[@]}" \
      --families ${families} \
      --resolutions ${RESOLUTIONS} \
      --checkpoint-roots ${checkpoint_roots} \
      --print-shards
  )
  if [[ "${#shards[@]}" -eq 0 ]]; then
    echo "No shards discovered for ${suite}." >&2
    exit 1
  fi

  shard_specs="$(join_words "${shards[@]}")"
  res2_specs=""
  other_specs=""
  for shard in "${shards[@]}"; do
    if [[ "${shard}" == *":2" ]]; then
      res2_specs="${res2_specs:+${res2_specs} }${shard}"
    else
      other_specs="${other_specs:+${other_specs} }${shard}"
    fi
  done

  echo "Suite ${suite}"
  echo "  families: ${families}"
  echo "  checkpoint roots: ${#roots[@]}"
  echo "  shards: ${shard_specs}"
  echo "  shard data dir: ${shard_data_dir}"

  if [[ -n "${res2_specs}" ]]; then
    job_id="$(submit_array_group "${res2_specs}" "${RES2_MEM}" "${RES2_TIME}" "${checkpoint_roots}" "${shard_data_dir}")"
    dependency_job_ids+=("${job_id}")
    echo "  submitted res2 eval job ${job_id}: ${res2_specs} (mem=${RES2_MEM}, time=${RES2_TIME})"
  fi
  if [[ -n "${other_specs}" ]]; then
    job_id="$(submit_array_group "${other_specs}" "${OTHER_MEM}" "${OTHER_TIME}" "${checkpoint_roots}" "${shard_data_dir}")"
    dependency_job_ids+=("${job_id}")
    echo "  submitted non-res2 eval job ${job_id}: ${other_specs} (mem=${OTHER_MEM}, time=${OTHER_TIME})"
  fi

  merge_dependency=$(IFS=:; echo "${dependency_job_ids[*]}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY RUN: sbatch --parsable --dependency=afterok:${merge_dependency} --export=ALL,SHARD_SPECS='${shard_specs}',SHARD_DATA_DIR='${shard_data_dir}',OUTPUT_DATA_DIR='${output_data_dir}',OUTPUT_IMAGE_DIR='${output_image_dir}',PLOT_PREFIX='${suite}',PLOT_LEAD_STEPS='${PLOT_LEAD_STEPS}' scripts/analyze_models/run_resolution_eval_merge.slurm" >&2
    merge_job_id="dry-merge-${suite}"
  else
    merge_job_id=$(
      sbatch \
        --parsable \
        --dependency="afterok:${merge_dependency}" \
        --export=ALL,SHARD_SPECS="${shard_specs}",SHARD_DATA_DIR="${shard_data_dir}",OUTPUT_DATA_DIR="${output_data_dir}",OUTPUT_IMAGE_DIR="${output_image_dir}",PLOT_PREFIX="${suite}",PLOT_LEAD_STEPS="${PLOT_LEAD_STEPS}" \
        scripts/analyze_models/run_resolution_eval_merge.slurm
    )
  fi
  echo "  submitted merge job ${merge_job_id}"
  echo "${merge_job_id}"
}

gc_merge_job="$(
  submit_suite "${GC_SUITE}" "graphcast gc_mamba" \
    "${VANILLA_ROOTS[@]}" \
    "${GC_MAMBA_ROOTS[@]}" \
  | tee /dev/stderr \
  | tail -n 1
)"

residual_merge_job="$(
  submit_suite "${RESIDUAL_SUITE}" "graphcast residual_mamba" \
    "${VANILLA_ROOTS[@]}" \
    "${RESIDUAL_MAMBA_ROOTS[@]}" \
  | tee /dev/stderr \
  | tail -n 1
)"

plot_dependency="${gc_merge_job}:${residual_merge_job}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY RUN: sbatch --parsable --dependency=afterok:${plot_dependency} --export=ALL,GC_MAMBA_CSV='${DATA_BASE}/${GC_SUITE}/resolution_eval.csv',RESIDUAL_MAMBA_CSV='${DATA_BASE}/${RESIDUAL_SUITE}/resolution_eval.csv',OUTPUT_IMAGE_DIR='${PLOT_IMAGE_DIR}',PLOT_LEAD_STEPS='${PLOT_LEAD_STEPS}' scripts/analyze_models/run_small_mamba_freeze_release_plot.slurm" >&2
  plot_job_id="dry-plot-small-mamba-freeze-release"
else
  plot_job_id=$(
    sbatch \
      --parsable \
      --dependency="afterok:${plot_dependency}" \
      --export=ALL,GC_MAMBA_CSV="${DATA_BASE}/${GC_SUITE}/resolution_eval.csv",RESIDUAL_MAMBA_CSV="${DATA_BASE}/${RESIDUAL_SUITE}/resolution_eval.csv",OUTPUT_IMAGE_DIR="${PLOT_IMAGE_DIR}",PLOT_LEAD_STEPS="${PLOT_LEAD_STEPS}" \
      scripts/analyze_models/run_small_mamba_freeze_release_plot.slurm
  )
fi

echo "Submitted final plot job: ${plot_job_id} after merge jobs ${plot_dependency}"
