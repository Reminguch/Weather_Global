#!/usr/bin/env bash
# Scalar-score checks and plotting; no spatial GPU fields retained.
#SBATCH --account=dabanin
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
set -eo pipefail
PHASE="${1:?phase required}"
WORKFLOW="${2:?workflow required}"
cd "${SLURM_SUBMIT_DIR}"
source scripts/graphcast_env.sh
export PYTHONPATH="${WORKFLOW}/source:${WORKFLOW}/source/third_party/graphcast"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/mpl-k22-${SLURM_JOB_ID}"
if [[ "${PHASE}" == readiness ]]; then
  python -u - "${WORKFLOW}" <<'PY'
import json, sys, time
from pathlib import Path
root=Path(sys.argv[1])
for _ in range(660):
    if (root/'staging_failed.json').exists():
        raise RuntimeError((root/'staging_failed.json').read_text())
    if (root/'staging_complete.json').exists():
        d=json.loads((root/'staging_complete.json').read_text())
        if d['status']!='complete' or d['inputs']!=69:
            raise ValueError('Staging is incomplete')
        print('staging ready', flush=True)
        break
    time.sleep(10)
else:
    raise TimeoutError('Staging did not complete in110minutes')
PY
else
  python -u - "${WORKFLOW}" <<'PY'
import sys,time
from pathlib import Path
root=Path(sys.argv[1])
for _ in range(660):
    if (root/'references_failed.json').exists():
        raise RuntimeError((root/'references_failed.json').read_text())
    if (root/'references_complete.json').exists():
        break
    time.sleep(10)
else:
    raise TimeoutError('Matched references incomplete after110minutes')
PY
  exec python -u "${WORKFLOW}/source/scripts/analyze_models/report_k22_common_grid_2022.py" \
    report --workflow "${WORKFLOW}"
fi
