#!/usr/bin/env bash
# Internet-enabled data staging on della-vis1; forecasting runs in Slurm.
set -euo pipefail
WORKFLOW="${1:?absolute workflow path required}"
shift
if [[ "$(hostname -s)" != "della-vis1" ]]; then
  echo 'Run this data download on della-vis1.' >&2
  exit 2
fi
cd /scratch/gpfs/DABANIN/iv9432/Weather_global
source scripts/graphcast_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${WORKFLOW}/source:${WORKFLOW}/source/third_party/graphcast"
DATA_ROOT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data_root"])' "${WORKFLOW}/protocol.json")"
exec nice -n 10 python -u "${WORKFLOW}/source/scripts/preprocessing/stage_common_grid_2023.py" \
  --output-root "${DATA_ROOT}" --manifest "${WORKFLOW}/initializations.json" --workers 8 --resume "$@"
