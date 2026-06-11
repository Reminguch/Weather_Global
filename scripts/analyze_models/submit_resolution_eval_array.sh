#!/bin/bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

source scripts/graphcast_env.sh

DEFAULT_CHECKPOINT_ROOTS=(
  "artifacts/checkpoints/graphcast_res1_stream"
  "artifacts/checkpoints/graphcast_res2_stream"
  "artifacts/checkpoints/graphcast_res4_stream"
  "artifacts/checkpoints/graphcast_res6_stream"
  "artifacts/checkpoints/graphcast_res8_stream"
  "artifacts/checkpoints/graphcast_res9_stream"
  "artifacts/checkpoints/graphcast_res12_stream"
  "artifacts/checkpoints/graphcast_res15_stream"
)

FAMILIES="${FAMILIES:-graphcast}"
RESOLUTIONS="${RESOLUTIONS:-1 2 4 6 8 9 12 15}"
WARMUP_STEPS="${WARMUP_STEPS:-24}"
TRUNK_STEPS="${TRUNK_STEPS:-32}"
METRICS="${METRICS:-rmse_k}"
EVAL_MODES="${EVAL_MODES:-cold}"
CHECKPOINT_ROOTS="${CHECKPOINT_ROOTS:-${DEFAULT_CHECKPOINT_ROOTS[*]}}"
CHECKPOINT_PATHS="${CHECKPOINT_PATHS:-}"
DATA_SOURCE="${DATA_SOURCE:-prepared_array}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-data/graphcast/graphcast/dataset/prepared_stream}"
STATS_DIR="${STATS_DIR:-data/graphcast/graphcast/stats}"
EVAL_YEAR="${EVAL_YEAR:-}"
RESIDUAL_EVAL_SEMANTICS="${RESIDUAL_EVAL_SEMANTICS:-rollout}"
RESIDUAL_AR_FEEDBACK="${RESIDUAL_AR_FEEDBACK:-baseline_plus_residual}"
METRIC_GRID_RESOLUTION="${METRIC_GRID_RESOLUTION:-18}"
METRIC_VARIABLES="${METRIC_VARIABLES:-}"
LEAD_STEPS="${LEAD_STEPS:-}"
RES1_MEM="${RES1_MEM:-30G}"
DEFAULT_ARRAY_TIME="${DEFAULT_ARRAY_TIME:-00:40:00}"

DISCOVER_CMD=(
  python -u scripts/analyze_models/unified_resolution_eval.py
  --families ${FAMILIES}
  --data-source "${DATA_SOURCE}"
  --prepared-data-root "${PREPARED_DATA_ROOT}"
  --print-shards
)
if [[ -n "${RESOLUTIONS}" ]]; then
  DISCOVER_CMD+=(--resolutions ${RESOLUTIONS})
fi
if [[ -n "${CHECKPOINT_ROOTS}" ]]; then
  DISCOVER_CMD+=(--checkpoint-roots ${CHECKPOINT_ROOTS})
fi
if [[ -n "${CHECKPOINT_PATHS}" ]]; then
  DISCOVER_CMD+=(--checkpoint-paths "${CHECKPOINT_PATHS}")
fi

mapfile -t SHARDS < <("${DISCOVER_CMD[@]}")
if [[ "${#SHARDS[@]}" -eq 0 ]]; then
  echo "No family:res shards discovered." >&2
  exit 1
fi

SHARD_SPECS="${SHARDS[*]}"
RES1_SHARDS=()
OTHER_SHARDS=()
for shard in "${SHARDS[@]}"; do
  if [[ "${shard}" == *":1" ]]; then
    RES1_SHARDS+=("${shard}")
  else
    OTHER_SHARDS+=("${shard}")
  fi
done

DEPENDENCY_JOB_IDS=()

if [[ "${#RES1_SHARDS[@]}" -gt 0 ]]; then
  RES1_SPECS="${RES1_SHARDS[*]}"
  RES1_ARRAY_MAX=$((${#RES1_SHARDS[@]} - 1))
  RES1_JOB_ID=$(
    sbatch \
      --parsable \
      --time="${DEFAULT_ARRAY_TIME}" \
      --mem="${RES1_MEM}" \
      --array="0-${RES1_ARRAY_MAX}" \
      --export=ALL,SHARD_SPECS="${RES1_SPECS}",WARMUP_STEPS="${WARMUP_STEPS}",TRUNK_STEPS="${TRUNK_STEPS}",METRICS="${METRICS}",METRIC_VARIABLES="${METRIC_VARIABLES}",EVAL_MODES="${EVAL_MODES}",CHECKPOINT_ROOTS="${CHECKPOINT_ROOTS}",CHECKPOINT_PATHS="${CHECKPOINT_PATHS}",DATA_SOURCE="${DATA_SOURCE}",PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT}",STATS_DIR="${STATS_DIR}",EVAL_YEAR="${EVAL_YEAR}",RESIDUAL_EVAL_SEMANTICS="${RESIDUAL_EVAL_SEMANTICS}",RESIDUAL_AR_FEEDBACK="${RESIDUAL_AR_FEEDBACK}",METRIC_GRID_RESOLUTION="${METRIC_GRID_RESOLUTION}",LEAD_STEPS="${LEAD_STEPS}" \
      scripts/analyze_models/run_resolution_eval_array.slurm
  )
  DEPENDENCY_JOB_IDS+=("${RES1_JOB_ID}")
  echo "Submitted res=1 job: ${RES1_JOB_ID} for shards: ${RES1_SPECS} (mem=${RES1_MEM})"
fi

if [[ "${#OTHER_SHARDS[@]}" -gt 0 ]]; then
  OTHER_SPECS="${OTHER_SHARDS[*]}"
  ARRAY_MAX=$((${#OTHER_SHARDS[@]} - 1))
  ARRAY_JOB_ID=$(
    sbatch \
      --parsable \
      --time="${DEFAULT_ARRAY_TIME}" \
      --array="0-${ARRAY_MAX}" \
      --export=ALL,SHARD_SPECS="${OTHER_SPECS}",WARMUP_STEPS="${WARMUP_STEPS}",TRUNK_STEPS="${TRUNK_STEPS}",METRICS="${METRICS}",METRIC_VARIABLES="${METRIC_VARIABLES}",EVAL_MODES="${EVAL_MODES}",CHECKPOINT_ROOTS="${CHECKPOINT_ROOTS}",CHECKPOINT_PATHS="${CHECKPOINT_PATHS}",DATA_SOURCE="${DATA_SOURCE}",PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT}",STATS_DIR="${STATS_DIR}",EVAL_YEAR="${EVAL_YEAR}",RESIDUAL_EVAL_SEMANTICS="${RESIDUAL_EVAL_SEMANTICS}",RESIDUAL_AR_FEEDBACK="${RESIDUAL_AR_FEEDBACK}",METRIC_GRID_RESOLUTION="${METRIC_GRID_RESOLUTION}",LEAD_STEPS="${LEAD_STEPS}" \
      scripts/analyze_models/run_resolution_eval_array.slurm
  )
  DEPENDENCY_JOB_IDS+=("${ARRAY_JOB_ID}")
  echo "Submitted array job: ${ARRAY_JOB_ID} for shards: ${OTHER_SPECS}"
fi

if [[ "${#DEPENDENCY_JOB_IDS[@]}" -eq 0 ]]; then
  echo "No shard jobs were submitted." >&2
  exit 1
fi

MERGE_DEPENDENCY=$(IFS=:; echo "${DEPENDENCY_JOB_IDS[*]}")

MERGE_JOB_ID=$(
  sbatch \
    --parsable \
    --dependency="afterok:${MERGE_DEPENDENCY}" \
    --export=ALL,SHARD_SPECS="${SHARD_SPECS}" \
    scripts/analyze_models/run_resolution_eval_merge.slurm
)
echo "Submitted merge job: ${MERGE_JOB_ID}"
