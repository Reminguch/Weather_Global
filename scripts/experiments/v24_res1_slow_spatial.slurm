#!/usr/bin/env bash
# Comparable 2,000-update branches: 16-23h; observed peak CPU RAM up to 235 GiB.
# Evaluation overrides: 16G/04:30:00 for three matched32 evaluations (~3h observed).
#SBATCH --job-name=v24-spatial-lr
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=28:00:00
#SBATCH --output=logs/v24_spatial_lr_%A_%a.out
#SBATCH --error=logs/v24_spatial_lr_%A_%a.err
set -euo pipefail
WORKSPACE="${3:-${SLURM_SUBMIT_DIR:?workspace required}}"
EXPERIMENT="${2:-${WORKSPACE}/artifacts/checkpoints/v24_Ilya/res1_batch4_slow_spatial_20260916}"
SOURCE_ROOT="${EXPERIMENT}/source"
if [[ ! -f "${SOURCE_ROOT}/snapshot.json" ]]; then
  echo "Prepare a complete source snapshot before submission; historical partial snapshots need explicit recovery: ${EXPERIMENT}" >&2
  exit 1
fi
source "${SOURCE_ROOT}/scripts/experiments/run_frozen_experiment.sh"
run_frozen_experiment "${WORKSPACE}" "${EXPERIMENT}" \
  "scripts/experiments/run_v24_res1_slow_spatial.py" "${1:?stage required}" "${SLURM_ARRAY_TASK_ID:-0}"
