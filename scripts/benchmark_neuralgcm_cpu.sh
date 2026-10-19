#!/usr/bin/env bash
set -euo pipefail
: "${NGCM_LOCAL_ENV:?Set this to the extracted node-local CPU runtime}"
NGCM_DATA_ROOT="/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global/data/neuralgcm/prepared/era5_2015_2023"
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH="${NGCM_LOCAL_ENV}/lib" SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
unset PYTHONPATH PYTHONHOME
ulimit -c 0
for NGCM_THREADS in 1 4; do
  if [[ "$NGCM_THREADS" == 1 ]]; then NGCM_CPUS=0; else NGCM_CPUS=1-4; fi
  env OMP_NUM_THREADS="$NGCM_THREADS" OPENBLAS_NUM_THREADS="$NGCM_THREADS" \
      MKL_NUM_THREADS="$NGCM_THREADS" NUMEXPR_NUM_THREADS="$NGCM_THREADS" \
      JAX_COMPILATION_CACHE_DIR="${NGCM_LOCAL_ENV}/jax-cache-${NGCM_THREADS}" \
      JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
    taskset -c "$NGCM_CPUS" "${NGCM_LOCAL_ENV}/bin/python3.11" -u "${NGCM_LOCAL_ENV}/run_download.py" \
      benchmark --root "$NGCM_DATA_ROOT" --benchmark-root "${NGCM_LOCAL_ENV}/benchmark-res2p8-${NGCM_THREADS}" \
      --frames 3 --resolutions res2p8 > "${NGCM_LOCAL_ENV}/benchmark-res2p8-${NGCM_THREADS}.log" 2>&1 &
done
wait
