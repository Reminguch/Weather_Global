#!/usr/bin/env bash
# Run only after both targeted GPU reruns succeed.
# Original 122-start report15642245 used435580K and24seconds; allow extra I/O
# for independent rescoring of360 retained prediction fields.
#SBATCH --account=dabanin
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
REPAIR_WORKFLOW="${1:?repair workflow required}"
cd "${SLURM_SUBMIT_DIR}"
source scripts/graphcast_env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg MPLCONFIGDIR="/tmp/mpl-truth-repair-report-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
python -u - "${REPAIR_WORKFLOW}" <<'PY'
from pathlib import Path
import hashlib
import json
import subprocess
import sys
workflow = Path(sys.argv[1]).resolve()
manifest = json.loads((workflow / 'report_source/manifest.json').read_text())
for item in manifest['files']:
    for path_key in ('working_path', 'snapshot_path'):
        with Path(item[path_key]).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != item['sha256']:
            raise ValueError(f'Report source changed: {item[path_key]}')
for script in ('scripts/analyze_models/report_common_grid_truth_repair.py',
               'scripts/analyze_models/validate_common_grid_truth_repair_report.py'):
    subprocess.run([sys.executable, '-u', script, '--workflow', str(workflow)], check=True)
print('Repaired full-122 report and independent validation completed.', flush=True)
PY
