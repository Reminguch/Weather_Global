#!/usr/bin/env bash
# Four threads on the internet-enabled staging node, all runtime files on scratch.
set -euo pipefail
TASK_ROOT=/scratch/gpfs/MENGDIW/sh4809/weatherforecast
export TMPDIR=$TASK_ROOT/Weather_Global_rebuild/.tmp
export XDG_CACHE_HOME=$TASK_ROOT/Weather_Global_rebuild/.cache
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$TASK_ROOT/Weather_Global_rebuild"
if [[ ${1:-pilot} == month ]]; then
  NGCM_MONTH_INDEX=${SLURM_ARRAY_TASK_ID:-0}
  NGCM_ARGUMENTS=("$@")
  for ((NGCM_ARG=1; NGCM_ARG<${#NGCM_ARGUMENTS[@]}; NGCM_ARG++)); do
    if [[ ${NGCM_ARGUMENTS[NGCM_ARG]} == --index ]]; then
      NGCM_MONTH_INDEX=${NGCM_ARGUMENTS[NGCM_ARG+1]:-invalid}
    fi
  done
  if [[ ! $NGCM_MONTH_INDEX =~ ^[0-9]+$ ]] || ((10#$NGCM_MONTH_INDEX > 107)); then
    echo 'Monthly index must be 0..107' >&2
    exit 2
  fi
  NGCM_MONTH_INDEX=$((10#$NGCM_MONTH_INDEX))
  NGCM_DATA=$TASK_ROOT/Weather_Global/data/neuralgcm/prepared/era5_2015_2023_restore_20261104
  mkdir -p "$NGCM_DATA/download_locks"
  printf -v NGCM_LOCK 'month-%03d.lock' "$NGCM_MONTH_INDEX"
  exec 9>"$NGCM_DATA/download_locks/$NGCM_LOCK"
  if [[ ${NGCM_MONTH_LOCK_NONBLOCK:-0} == 1 ]]; then
    flock -n -E 75 9 || exit $?
  else
    flock 9
  fi
  printf -v NGCM_MONTH '%04d-%02d' "$((2015+NGCM_MONTH_INDEX/12))" "$((1+NGCM_MONTH_INDEX%12))"
  if [[ -f $NGCM_DATA/shards/$NGCM_MONTH/res2p8/READY.json && -f $NGCM_DATA/shards/$NGCM_MONTH/res2p8/profile.json ]]; then
    echo "Already verified $NGCM_MONTH; skipping download"
    exit 0
  fi
fi
exec "$TASK_ROOT/Weather_Global/.venv_ngcm_data/bin/python" -u scripts/preprocessing/restore_ngcm_2p8.py "${1:-pilot}" --root "$TASK_ROOT/Weather_Global/data/neuralgcm/prepared/era5_2015_2023_restore_20261104" --checkpoint "$TASK_ROOT/Weather_Global/data/neuralgcm/checkpoints/deterministic_2_8_deg.pkl" "${@:2}"
