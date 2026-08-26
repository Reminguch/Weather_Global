#!/bin/bash

set -euo pipefail

if [[ "$(hostname -s)" != "della-vis1" && "${ALLOW_NON_VIS1:-0}" != "1" ]]; then
  echo "This Internet-enabled build must run on della-vis1. Set ALLOW_NON_VIS1=1 only for a brief smoke." >&2
  exit 2
fi

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
source scripts/graphcast_env.sh

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

exec nice -n 10 python -u scripts/build_graphcast37_prepared_stream.py \
  --start-year 2015 \
  --end-year 2022 \
  --out-root data/graphcast/graphcast/dataset/prepared_stream_graphcast37 \
  --temp-root data/graphcast/graphcast/dataset/.tmp_graphcast37_staging \
  --chunk-time 4 \
  --staging-chunk-time 16 \
  --max-workers 8 \
  --resume \
  --extend-existing \
  --delete-staged \
  --min-free-tib 6 \
  "$@"
