#!/usr/bin/env bash
# Build the native v24 early SWA from the first four retained uniform-sparse
# res0.25 checkpoints.  This is parameter-only SWA; evaluation uses the
# canonical cold, zero recurrent state policy.
#SBATCH --job-name=v24-r025-spunswa
#SBATCH --account=dabanin
#SBATCH --qos=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/v24_r025_spuni_early_swa_%j.out
#SBATCH --error=logs/v24_r025_spuni_early_swa_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

RUN_DIR="artifacts/checkpoints/v24_Ilya/res0p25_sparse_uniform_weight_ablation_20260829/legacy_di16_bcg1_sparse_uniform_h20_no_tzero_fp32_lr3em6_sg500"
SWA_TAG="swa_step00050-00200"
OUT="${RUN_DIR}/swa/${SWA_TAG}.pkl"
INPUTS=(
  "${RUN_DIR}/checkpoints/checkpoint_step00000050.pkl"
  "${RUN_DIR}/checkpoints/checkpoint_step00000100.pkl"
  "${RUN_DIR}/checkpoints/checkpoint_step00000150.pkl"
  "${RUN_DIR}/checkpoints/checkpoint_step00000200.pkl"
)
SOURCE_STEPS=(50 100 150 200)

for required in "${INPUTS[@]}"; do
  [[ -f "${required}" ]] || { echo "Missing SWA input: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUT}" ]] || { echo "Refusing to overwrite existing SWA output: ${OUT}" >&2; exit 1; }
mkdir -p "$(dirname "${OUT}")"

export JAX_PLATFORMS=cpu
source scripts/graphcast_env.sh
python -u scripts/training/build_v24_Ilya_swa.py \
  --inputs "${INPUTS[@]}" \
  --source-steps "${SOURCE_STEPS[@]}" \
  --output "${OUT}"

echo "Completed ${SWA_TAG}: ${OUT}"
