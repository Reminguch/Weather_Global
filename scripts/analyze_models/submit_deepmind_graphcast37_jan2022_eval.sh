#!/bin/bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

source scripts/graphcast_env.sh

GRAPHCAST37_PARAMS="${GRAPHCAST37_PARAMS:-}"
GCSMALL_PARAMS="${GCSMALL_PARAMS:-data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz}"
GRAPHCAST37_PREPARED_ROOT="${GRAPHCAST37_PREPARED_ROOT:-data/graphcast/graphcast/dataset/prepared_stream_graphcast37_jan2022}"
GCSMALL_PREPARED_ROOT="${GCSMALL_PREPARED_ROOT:-data/graphcast/graphcast/dataset/prepared_stream_gcsmall_jan2022}"
GRAPHCAST37_STATS_DIR="${GRAPHCAST37_STATS_DIR:-data/graphcast/graphcast/stats_graphcast_37}"
GCSMALL_STATS_DIR="${GCSMALL_STATS_DIR:-data/graphcast/graphcast/stats}"
OUTPUT_ROOT="${OUTPUT_ROOT:-plots/analyze_models/data/resolution_eval/deepmind_graphcast37_res0p25_vs_gcsmall_res1_jan2022}"
IMAGE_ROOT="${IMAGE_ROOT:-plots/analyze_models/images/resolution_eval/deepmind_graphcast37_res0p25_vs_gcsmall_res1_jan2022}"
LEAD_STEPS="${LEAD_STEPS:-4 16 24 32 40}"
EVAL_START="${EVAL_START:-2022-01-01 00:00:00}"
EVAL_END="${EVAL_END:-2022-02-01 00:00:00}"
METRIC_GRID_RESOLUTION="${METRIC_GRID_RESOLUTION:-1}"
METRICS="${METRICS:-rmse_k}"
METRIC_VARIABLES="${METRIC_VARIABLES:-2m_temperature}"
EVAL_MODES="${EVAL_MODES:-cold}"
WINDOW_BATCH_SIZE="${WINDOW_BATCH_SIZE:-1}"
RES1_WINDOW_BATCH_SIZE="${RES1_WINDOW_BATCH_SIZE:-1}"
GRAPHCAST37_TIME="${GRAPHCAST37_TIME:-06:00:00}"
GRAPHCAST37_MEM="${GRAPHCAST37_MEM:-180G}"
GCSMALL_TIME="${GCSMALL_TIME:-01:00:00}"
GCSMALL_MEM="${GCSMALL_MEM:-40G}"

if [[ -z "${GRAPHCAST37_PARAMS}" ]]; then
  mapfile -t candidates < <(find data/graphcast/graphcast/params -maxdepth 1 -type f -name 'GraphCast*resolution 0.25*pressure levels 37*.npz' | sort)
  if [[ "${#candidates[@]}" -eq 1 ]]; then
    GRAPHCAST37_PARAMS="${candidates[0]}"
  else
    echo "Set GRAPHCAST37_PARAMS to the DeepMind GraphCast 0.25-degree, 37-level params file." >&2
    exit 1
  fi
fi

for required in "${GRAPHCAST37_PARAMS}" "${GCSMALL_PARAMS}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing params file: ${required}" >&2
    exit 1
  fi
done
for required_dir in "${GRAPHCAST37_PREPARED_ROOT}/res0p25" "${GCSMALL_PREPARED_ROOT}/res1" "${GRAPHCAST37_STATS_DIR}" "${GCSMALL_STATS_DIR}"; do
  if [[ ! -e "${required_dir}" ]]; then
    echo "Missing required path: ${required_dir}" >&2
    exit 1
  fi
done

SHARD_ROOT="${OUTPUT_ROOT}/shards"
mkdir -p "${SHARD_ROOT}" "${IMAGE_ROOT}"

GRAPHCAST37_JOB_ID=$(
  sbatch \
    --parsable \
    --time="${GRAPHCAST37_TIME}" \
    --mem="${GRAPHCAST37_MEM}" \
    --array=0-0 \
    --export=ALL,SHARD_SPECS="graphcast:0.25",DATA_SOURCE=prepared_array,PREPARED_DATA_ROOT="${GRAPHCAST37_PREPARED_ROOT}",STATS_DIR="${GRAPHCAST37_STATS_DIR}",CHECKPOINT_PATHS="${GRAPHCAST37_PARAMS}",EVAL_START="${EVAL_START}",EVAL_END="${EVAL_END}",LEAD_STEPS="${LEAD_STEPS}",METRICS="${METRICS}",METRIC_VARIABLES="${METRIC_VARIABLES}",EVAL_MODES="${EVAL_MODES}",METRIC_GRID_RESOLUTION="${METRIC_GRID_RESOLUTION}",WINDOW_BATCH_SIZE="${WINDOW_BATCH_SIZE}",RES1_WINDOW_BATCH_SIZE="${RES1_WINDOW_BATCH_SIZE}",SHARD_DATA_DIR="${SHARD_ROOT}",OUTPUT_CSV_SUFFIX="_graphcast37_jan2022" \
    scripts/analyze_models/run_resolution_eval_array.slurm
)
echo "Submitted GraphCast37 eval job: ${GRAPHCAST37_JOB_ID}"

GCSMALL_JOB_ID=$(
  sbatch \
    --parsable \
    --time="${GCSMALL_TIME}" \
    --mem="${GCSMALL_MEM}" \
    --array=0-0 \
    --export=ALL,SHARD_SPECS="graphcast:1",DATA_SOURCE=prepared_array,PREPARED_DATA_ROOT="${GCSMALL_PREPARED_ROOT}",STATS_DIR="${GCSMALL_STATS_DIR}",CHECKPOINT_PATHS="${GCSMALL_PARAMS}",EVAL_START="${EVAL_START}",EVAL_END="${EVAL_END}",LEAD_STEPS="${LEAD_STEPS}",METRICS="${METRICS}",METRIC_VARIABLES="${METRIC_VARIABLES}",EVAL_MODES="${EVAL_MODES}",METRIC_GRID_RESOLUTION="${METRIC_GRID_RESOLUTION}",WINDOW_BATCH_SIZE="${WINDOW_BATCH_SIZE}",RES1_WINDOW_BATCH_SIZE="${RES1_WINDOW_BATCH_SIZE}",SHARD_DATA_DIR="${SHARD_ROOT}",OUTPUT_CSV_SUFFIX="_gcsmall_jan2022" \
    scripts/analyze_models/run_resolution_eval_array.slurm
)
echo "Submitted GC_small eval job: ${GCSMALL_JOB_ID}"

MERGE_JOB_ID=$(
  sbatch \
    --parsable \
    --dependency="afterok:${GRAPHCAST37_JOB_ID}:${GCSMALL_JOB_ID}" \
    --export=ALL,SHARD_SPECS="graphcast:0.25 graphcast:1",SHARD_DATA_DIR="${SHARD_ROOT}",OUTPUT_DATA_DIR="${OUTPUT_ROOT}",OUTPUT_IMAGE_DIR="${IMAGE_ROOT}",PLOT_PREFIX="deepmind_graphcast37_vs_gcsmall_jan2022",PLOT_LEAD_STEPS="${LEAD_STEPS}",BASELINE_RES=1 \
    scripts/analyze_models/run_resolution_eval_merge.slurm
)
echo "Submitted merge job: ${MERGE_JOB_ID}"
