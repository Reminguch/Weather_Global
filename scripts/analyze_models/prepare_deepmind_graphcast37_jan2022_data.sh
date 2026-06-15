#!/bin/bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

source scripts/graphcast_env.sh

GRAPHCAST37_DATA="${GRAPHCAST37_DATA:-data/graphcast/graphcast/dataset/wb2_graphcast37_jan2022_0p25.zarr}"
GRAPHCAST37_PREPARED_ROOT="${GRAPHCAST37_PREPARED_ROOT:-data/graphcast/graphcast/dataset/prepared_stream_graphcast37_jan2022}"
GCSMALL_DATA="${GCSMALL_DATA:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}"
GCSMALL_PREPARED_ROOT="${GCSMALL_PREPARED_ROOT:-data/graphcast/graphcast/dataset/prepared_stream_gcsmall_jan2022}"
GRAPHCAST37_STATS_DIR="${GRAPHCAST37_STATS_DIR:-data/graphcast/graphcast/stats_graphcast_37}"
GRAPHCAST37_PARAMS="${GRAPHCAST37_PARAMS:-}"
GCSMALL_PARAMS="${GCSMALL_PARAMS:-data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz}"
WINDOW_START="${WINDOW_START:-2021-12-31 18:00:00}"
WINDOW_END="${WINDOW_END:-2022-02-10 18:00:00}"
OVERWRITE="${OVERWRITE:-0}"

DOWNLOAD_ARGS=(
  python -u scripts/download_deepmind_graphcast_assets.py
  --stats-dir "${GRAPHCAST37_STATS_DIR}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  DOWNLOAD_ARGS+=(--overwrite)
fi
if [[ -n "${GRAPHCAST37_PARAMS_NAME:-}" ]]; then
  DOWNLOAD_ARGS+=(--params-name "${GRAPHCAST37_PARAMS_NAME}")
fi
echo "Running: ${DOWNLOAD_ARGS[*]}"
"${DOWNLOAD_ARGS[@]}"

if [[ -z "${GRAPHCAST37_PARAMS}" ]]; then
  mapfile -t candidates < <(find data/graphcast/graphcast/params -maxdepth 1 -type f -name 'GraphCast*resolution 0.25*pressure levels 37*.npz' | sort)
  if [[ "${#candidates[@]}" -eq 1 ]]; then
    GRAPHCAST37_PARAMS="${candidates[0]}"
  else
    echo "Set GRAPHCAST37_PARAMS after downloading GraphCast37 params." >&2
    exit 1
  fi
fi

STAGE_ARGS=(
  python -u scripts/stage_wb2_graphcast37_window.py
  --output "${GRAPHCAST37_DATA}"
  --start-time "${WINDOW_START}"
  --end-time "${WINDOW_END}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  STAGE_ARGS+=(--overwrite)
fi
echo "Running: ${STAGE_ARGS[*]}"
"${STAGE_ARGS[@]}"

PREP_GRAPHCAST37=(
  python -u -m src.data_operations.preprocessing.prepare_graphcast_streaming_store
  --data-path "${GRAPHCAST37_DATA}"
  --ckpt-in "${GRAPHCAST37_PARAMS}"
  --out-root "${GRAPHCAST37_PREPARED_ROOT}"
  --resolutions 0.25 1
  --time-start "${WINDOW_START}"
  --time-end "${WINDOW_END}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  PREP_GRAPHCAST37+=(--overwrite)
fi
echo "Running: ${PREP_GRAPHCAST37[*]}"
"${PREP_GRAPHCAST37[@]}"

PREP_GCSMALL=(
  python -u -m src.data_operations.preprocessing.prepare_graphcast_streaming_store
  --data-path "${GCSMALL_DATA}"
  --ckpt-in "${GCSMALL_PARAMS}"
  --out-root "${GCSMALL_PREPARED_ROOT}"
  --resolutions 1
  --time-start "${WINDOW_START}"
  --time-end "${WINDOW_END}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  PREP_GCSMALL+=(--overwrite)
fi
echo "Running: ${PREP_GCSMALL[*]}"
"${PREP_GCSMALL[@]}"
