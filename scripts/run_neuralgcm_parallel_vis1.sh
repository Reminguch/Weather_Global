#!/usr/bin/env bash
# Resume 2.8-degree ERA5 using the verified, self-contained local CPU runtime.
set -euo pipefail
if [[ "$(hostname -s)" != "della-vis1" ]]; then
  echo "Run this launcher on della-vis1." >&2
  exit 2
fi
NGCM_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NGCM_DATA_ROOT="/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global/data/neuralgcm/prepared/era5_2015_2023"
NGCM_LOCAL_ENV="${NGCM_LOCAL_ENV:-/tmp/sh4809-neuralgcm-cpu}"
if [[ ! -x "${NGCM_LOCAL_ENV}/bin/python3.11" || ! -f "${NGCM_LOCAL_ENV}/run_download.py" ]]; then
  echo "Missing complete CPU runtime. Build with scripts/preprocessing/build_neuralgcm_cpu_runtime.py" >&2
  echo "Extract it on this node and set NGCM_LOCAL_ENV to that directory." >&2
  exit 2
fi
cd "${NGCM_LOCAL_ENV}"
ulimit -c 0

# Check fileset-visible space before resuming. The initial quota audit also
# checked the project byte and inode limits; df alone is not a quota audit.
/usr/bin/python3 - "${NGCM_DATA_ROOT}" <<'PY'
import pathlib, shutil, sys
root = pathlib.Path(sys.argv[1])
if not (root / 'download_manifest.json').is_file():
    raise SystemExit('Missing migrated download manifest')
free = shutil.disk_usage(root).free
if free < 128 * 1024**3:
    raise SystemExit('Need at least 128 GiB available before resuming 2.8-degree downloads')
print(f'Scratch available: {free / 1024**4:.2f} TiB', flush=True)
PY

export LC_ALL=C JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
export LD_LIBRARY_PATH="${NGCM_LOCAL_ENV}/lib"
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
NGCM_DOWNLOAD_WORKERS="${NGCM_DOWNLOAD_WORKERS:-20}"
NGCM_THREADS_PER_WORKER="${NGCM_THREADS_PER_WORKER:-4}"
export OMP_NUM_THREADS="${NGCM_THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${NGCM_THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${NGCM_THREADS_PER_WORKER}"
export NUMEXPR_NUM_THREADS="${NGCM_THREADS_PER_WORKER}"
# Reuse the in-process JIT cache across months. JAX 0.9.2's persisted CPU
# executables emitted target-feature mismatch warnings even on this same node.
export JAX_ENABLE_COMPILATION_CACHE=false
unset JAX_COMPILATION_CACHE_DIR JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS
exec "${NGCM_LOCAL_ENV}/bin/python3.11" -u "${NGCM_LOCAL_ENV}/run_download.py" supervise \
  --root "${NGCM_DATA_ROOT}" \
  --workers "${NGCM_DOWNLOAD_WORKERS}" \
  --threads-per-worker "${NGCM_THREADS_PER_WORKER}" \
  --resolutions res2p8
