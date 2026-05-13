#!/bin/bash
#SBATCH --job-name=prepare-gc-stream
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prepare_graphcast_stream_%j.out
#SBATCH --error=logs/prepare_graphcast_stream_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs

source scripts/graphcast_env.sh

DATA_PATH="${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}"
CKPT_IN="${CKPT_IN:-data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz}"
OUT_ROOT="${OUT_ROOT:-data/graphcast/graphcast/dataset/prepared_stream}"
RESOLUTIONS="${RESOLUTIONS:-1}"
OVERWRITE="${OVERWRITE:-0}"

CMD=(
  python -u -m src.data_operations.preprocessing.prepare_graphcast_streaming_store
  --data-path "${DATA_PATH}"
  --ckpt-in "${CKPT_IN}"
  --out-root "${OUT_ROOT}"
  --resolutions ${RESOLUTIONS}
)
if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
