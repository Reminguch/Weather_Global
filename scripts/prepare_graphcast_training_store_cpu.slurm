#!/bin/bash
#SBATCH --job-name=gc-prepare
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/gc_prepare_%A_%a.out
#SBATCH --error=logs/gc_prepare_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
source scripts/graphcast_env.sh

DATA_PATH=${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}
OUT_ROOT=${OUT_ROOT:-data/graphcast/graphcast/dataset/prepared}
CKPT_IN=${CKPT_IN:-"data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"}
OVERWRITE=${OVERWRITE:-0}

RESOLUTIONS=(${RESOLUTIONS:-1.0 2.0 4.0 9.0 15.0})
RESOLUTION="${RESOLUTIONS[${SLURM_ARRAY_TASK_ID}]}"

CMD=(
  python -u -m src.data_operations.preprocessing.prepare_graphcast_training_store
  --data-path "${DATA_PATH}"
  --ckpt-in "${CKPT_IN}"
  --out-root "${OUT_ROOT}"
  --resolutions "${RESOLUTION}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "Preparing GraphCast training store at resolution ${RESOLUTION}"
echo "Data path: ${DATA_PATH}"
echo "Output root: ${OUT_ROOT}"
"${CMD[@]}"
