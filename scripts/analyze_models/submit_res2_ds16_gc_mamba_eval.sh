#!/bin/bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

EXPERIMENT="res2_ds16_gc_mamba_target_steps_bptt16_warm_leads1_9d"
DATA_DIR="plots/analyze_models/data/resolution_eval/${EXPERIMENT}"
SHARD_DIR="${DATA_DIR}/shards"
IMAGE_DIR="plots/analyze_models/images/resolution_eval/${EXPERIMENT}"
LEAD_STEPS="4 8 12 16 20 24 28 32 36"
METRICS="weighted_allvars rmse_k"
EVAL_MODES="warm"
MEM="${RES2_DS16_EVAL_MEM:-64G}"
TIME="${RES2_DS16_EVAL_TIME:-02:00:00}"

mkdir -p "${SHARD_DIR}" "${IMAGE_DIR}" logs

MAMBA_ROOT="artifacts/checkpoints/7_years/mamba_frozen_from_vanilla_mp6_20k_res2_target_steps_bptt16"
declare -a EVAL_SPECS=(
  "gc_mamba gc_di16_ds16_ts4 ${MAMBA_ROOT}/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di16_ds16_20k_target_step4_bptt16/ckpt_best.npz"
  "gc_mamba gc_di16_ds16_ts8 ${MAMBA_ROOT}/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di16_ds16_20k_target_step8_bptt16/ckpt_best.npz"
  "gc_mamba gc_di16_ds16_ts12 ${MAMBA_ROOT}/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di16_ds16_20k_target_step12_bptt16/ckpt_best.npz"
  "gc_mamba gc_di64_ds16_ts4 ${MAMBA_ROOT}/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di64_ds16_20k_target_step4_bptt16/ckpt_best.npz"
  "gc_mamba gc_di64_ds16_ts8 ${MAMBA_ROOT}/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di64_ds16_20k_target_step8_bptt16/ckpt_best.npz"
  "gc_mamba gc_di64_ds16_ts12 ${MAMBA_ROOT}/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di64_ds16_20k_target_step12_bptt16/ckpt_best.npz"
  "gc_mamba gc_di256_ds16_ts4 ${MAMBA_ROOT}/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di256_ds16_20k_target_step4_bptt16/ckpt_best.npz"
  "gc_mamba gc_di256_ds16_ts8 ${MAMBA_ROOT}/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di256_ds16_20k_target_step8_bptt16/ckpt_best.npz"
  "gc_mamba gc_di256_ds16_ts12 ${MAMBA_ROOT}/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di256_ds16_20k_target_step12_bptt16/ckpt_best.npz"
  "graphcast baseline_init artifacts/checkpoints/7_years/vanilla_gc/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k/ckpt_best.npz"
  "graphcast baseline_continue20k artifacts/checkpoints/7_years/vanilla_gc_mp6_continue20k/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_continue20k/ckpt_best.npz"
)

for spec in "${EVAL_SPECS[@]}"; do
  read -r family label ckpt <<< "${spec}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint for ${label}: ${ckpt}" >&2
    exit 1
  fi
done

if [[ "${#EVAL_SPECS[@]}" -ne 11 ]]; then
  echo "Expected 11 eval specs, got ${#EVAL_SPECS[@]}" >&2
  exit 1
fi

echo "Experiment: ${EXPERIMENT}"
echo "Shard dir: ${SHARD_DIR}"
echo "Lead steps: ${LEAD_STEPS}"
echo "Eval specs:"
printf '  %s\n' "${EVAL_SPECS[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run only; no jobs submitted."
  exit 0
fi

JOB_IDS=()
for spec in "${EVAL_SPECS[@]}"; do
  read -r family label ckpt <<< "${spec}"
  job_id=$(
    sbatch \
      --parsable \
      --job-name="r2ds16-${label}" \
      --time="${TIME}" \
      --mem="${MEM}" \
      --array=0-0 \
      --export=ALL,SHARD_SPECS="${family}:2",WARMUP_STEPS=24,TRUNK_STEPS=32,METRICS="${METRICS}",EVAL_MODES="${EVAL_MODES}",CHECKPOINT_ROOTS="",CHECKPOINT_PATHS="${ckpt}",DATA_SOURCE=prepared_array,PREPARED_DATA_ROOT=data/graphcast/graphcast/dataset/prepared_stream,STATS_DIR=data/graphcast/graphcast/stats,METRIC_GRID_RESOLUTION=2,LEAD_STEPS="${LEAD_STEPS}",SHARD_DATA_DIR="${SHARD_DIR}",OUTPUT_CSV_SUFFIX="_${label}",WINDOW_BATCH_SIZE=1,PREPARED_LOAD_WORKERS=2 \
      scripts/analyze_models/run_resolution_eval_array.slurm
  )
  JOB_IDS+=("${job_id}")
  echo "Submitted ${label}: ${job_id}"
done

DEPENDENCY=$(IFS=:; echo "${JOB_IDS[*]}")
plot_job_id=$(
  sbatch \
    --parsable \
    --dependency="afterok:${DEPENDENCY}" \
    --export=ALL,EXPERIMENT="${EXPERIMENT}",DATA_DIR="${DATA_DIR}",SHARD_DIR="${SHARD_DIR}",IMAGE_DIR="${IMAGE_DIR}",LEAD_STEPS="${LEAD_STEPS}" \
    scripts/analyze_models/run_res2_ds16_merge_plot.slurm
)
echo "Submitted merge/plot job: ${plot_job_id}"

ny_job_id=$(
  sbatch \
    --parsable \
    --array=0-2 \
    --export=ALL,LEAD_DAYS_LIST="2 4 6",NY_BATCH_SIZE=8 \
    scripts/analyze_models/run_res2_ds16_ny_trajectory.slurm
)
echo "Submitted NY trajectory job array: ${ny_job_id}"
echo "Monitor: squeue -j ${DEPENDENCY}:${plot_job_id}:${ny_job_id}"
