#!/bin/bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

DRY_RUN=${DRY_RUN:-0}
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

EXPERIMENT="res2_ds16_gc_mamba_k12_gclr3e7_mlr1e5_tdrop10_release_best_final_warm_leads1_9d"
DATA_DIR="plots/analyze_models/data/resolution_eval/${EXPERIMENT}"
SHARD_DIR="${DATA_DIR}/shards"
IMAGE_DIR="plots/analyze_models/images/resolution_eval/${EXPERIMENT}"
LEAD_STEPS="${LEAD_STEPS:-4 8 12 16 20 24 28 32 36}"
METRICS="${METRICS:-weighted_allvars rmse_k}"
EVAL_MODES="${EVAL_MODES:-warm}"
MEM="${RES2_DS16_K12_TDROP10_EVAL_MEM:-48G}"
TIME="${RES2_DS16_K12_TDROP10_EVAL_TIME:-01:30:00}"
UPSTREAM_DEPENDENCY="${UPSTREAM_DEPENDENCY:-}"

mkdir -p "${SHARD_DIR}" "${IMAGE_DIR}" logs

FROZEN_ROOT="artifacts/checkpoints/7_years/mamba_frozen_from_vanilla_mp6_20k_res2_target_steps_bptt16"
NODROP_ROOT="artifacts/checkpoints/7_years/mamba_release_all_from_frozen20k_res2_ds16_k12_gclr1e6_mlr1e5_20k"
TDROP_ROOT="artifacts/checkpoints/7_years/mamba_release_all_from_frozen20k_res2_ds16_k12_gclr3e7_mlr1e5_tdrop10_20k"
BASE_NAME="vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k"

declare -a EVAL_SPECS=()
for d_inner in 16 64; do
  run="${BASE_NAME}_gc_mamba_tc2_di${d_inner}_ds16_20k_target_step12_bptt16"
  tdrop_run="${run}_release_all20k_gclr3e7_mlr1e5_tdrop0p10"
  nodrop_run="${run}_release_all20k_gclr1e6_mlr1e5"
  EVAL_SPECS+=(
    "gc_mamba tdrop_di${d_inner}_best ${TDROP_ROOT}/${tdrop_run}/ckpt_best.npz"
    "gc_mamba tdrop_di${d_inner}_step40000 ${TDROP_ROOT}/${tdrop_run}/ckpt_step40000.npz"
    "gc_mamba nodrop_di${d_inner}_best ${NODROP_ROOT}/${nodrop_run}/ckpt_best.npz"
    "gc_mamba nodrop_di${d_inner}_step40000 ${NODROP_ROOT}/${nodrop_run}/ckpt_step40000.npz"
    "gc_mamba frozen_di${d_inner}_best ${FROZEN_ROOT}/${run}/ckpt_best.npz"
  )
done
EVAL_SPECS+=(
  "graphcast baseline_init artifacts/checkpoints/7_years/vanilla_gc/${BASE_NAME}/ckpt_best.npz"
  "graphcast baseline_continue20k artifacts/checkpoints/7_years/vanilla_gc_mp6_continue20k/${BASE_NAME}_continue20k/ckpt_best.npz"
)

if [[ "${#EVAL_SPECS[@]}" -ne 12 ]]; then
  echo "Expected 12 eval specs, got ${#EVAL_SPECS[@]}" >&2
  exit 1
fi

for spec in "${EVAL_SPECS[@]}"; do
  read -r family label ckpt <<< "${spec}"
  if [[ ! -f "${ckpt}" ]]; then
    if [[ -n "${UPSTREAM_DEPENDENCY}" && "${label}" == tdrop_* ]]; then
      echo "Deferring missing tdrop checkpoint for ${label} until after dependency ${UPSTREAM_DEPENDENCY}: ${ckpt}" >&2
    elif [[ "${DRY_RUN}" == "1" ]]; then
      echo "DRY RUN: missing checkpoint for ${label}: ${ckpt}" >&2
    else
      echo "Missing checkpoint for ${label}: ${ckpt}" >&2
      exit 1
    fi
  fi
done

echo "Experiment: ${EXPERIMENT}"
echo "Shard dir: ${SHARD_DIR}"
echo "Lead steps: ${LEAD_STEPS}"
echo "Metrics: ${METRICS}"
echo "Eval modes: ${EVAL_MODES}"
echo "Eval resources: mem=${MEM} time=${TIME}"
if [[ -n "${UPSTREAM_DEPENDENCY}" ]]; then
  echo "Eval dependency: afterok:${UPSTREAM_DEPENDENCY}"
fi
echo "Eval specs (${#EVAL_SPECS[@]}):"
printf '  %s\n' "${EVAL_SPECS[@]}"

JOB_IDS=()
for spec in "${EVAL_SPECS[@]}"; do
  read -r family label ckpt <<< "${spec}"
  SBATCH_CMD=(
    sbatch
    --parsable
    --job-name="r2k12td10-${label}"
    --time="${TIME}"
    --mem="${MEM}"
    --array=0-0
    --export=ALL,SHARD_SPECS="${family}:2",WARMUP_STEPS=24,TRUNK_STEPS=32,METRICS="${METRICS}",EVAL_MODES="${EVAL_MODES}",CHECKPOINT_ROOTS="",CHECKPOINT_PATHS="${ckpt}",DATA_SOURCE=prepared_array,PREPARED_DATA_ROOT=data/graphcast/graphcast/dataset/prepared_stream,STATS_DIR=data/graphcast/graphcast/stats,METRIC_GRID_RESOLUTION=2,LEAD_STEPS="${LEAD_STEPS}",SHARD_DATA_DIR="${SHARD_DIR}",OUTPUT_CSV_SUFFIX="_${label}",WINDOW_BATCH_SIZE=1,PREPARED_LOAD_WORKERS=2
  )
  if [[ -n "${UPSTREAM_DEPENDENCY}" ]]; then
    SBATCH_CMD+=(--dependency="afterok:${UPSTREAM_DEPENDENCY}")
  fi
  SBATCH_CMD+=(scripts/analyze_models/run_resolution_eval_array.slurm)

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY RUN eval %s:' "${label}"
    printf ' %q' "${SBATCH_CMD[@]}"
    printf '\n'
  else
    job_raw=$("${SBATCH_CMD[@]}")
    job_id="${job_raw%%;*}"
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
      echo "Eval sbatch did not return a valid job id for ${label}: ${job_raw}" >&2
      exit 1
    fi
    JOB_IDS+=("${job_id}")
    echo "Submitted ${label}: ${job_id}"
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run only; no jobs submitted."
  exit 0
fi

DEPENDENCY=$(IFS=:; echo "${JOB_IDS[*]}")
plot_job_id=$(
  sbatch \
    --parsable \
    --dependency="afterok:${DEPENDENCY}" \
    --export=ALL,EXPERIMENT="${EXPERIMENT}",DATA_DIR="${DATA_DIR}",SHARD_DIR="${SHARD_DIR}",IMAGE_DIR="${IMAGE_DIR}",PLOT_PREFIX="${EXPERIMENT}",LEAD_STEPS="${LEAD_STEPS}" \
    scripts/analyze_models/run_res2_ds16_merge_plot.slurm
)
plot_job_id="${plot_job_id%%;*}"
if [[ ! "${plot_job_id}" =~ ^[0-9]+$ ]]; then
  echo "Merge/plot sbatch did not return a valid job id: ${plot_job_id}" >&2
  exit 1
fi
echo "Submitted merge/plot job: ${plot_job_id}"

MONITOR_IDS="${DEPENDENCY//:/,},${plot_job_id}"
if [[ -n "${UPSTREAM_DEPENDENCY}" ]]; then
  echo "Monitor: squeue -j ${UPSTREAM_DEPENDENCY},${MONITOR_IDS}"
else
  echo "Monitor: squeue -j ${MONITOR_IDS}"
fi
