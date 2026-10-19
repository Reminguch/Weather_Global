#!/usr/bin/env bash
# Resources are overridden per stage after the batch-4 GPU smoke measurement.
#SBATCH --job-name=v24-r1-b4lr
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=01:02:00
#SBATCH --output=logs/v24_r1_b4lr_%A_%a.out
#SBATCH --error=logs/v24_r1_b4lr_%A_%a.err
set -euo pipefail
WORKSPACE="${3:-${SLURM_SUBMIT_DIR:?workspace required}}"
EXPERIMENT="${2:-${WORKSPACE}/artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908}"
SOURCE_ROOT="${EXPERIMENT}/source"
if [[ ! -f "${SOURCE_ROOT}/snapshot.json" ]]; then
  echo "Prepare a complete source snapshot before submission; historical partial snapshots need explicit recovery: ${EXPERIMENT}" >&2
  exit 1
fi
source "${SOURCE_ROOT}/scripts/experiments/run_frozen_experiment.sh"
run_frozen_experiment "${WORKSPACE}" "${EXPERIMENT}" \
  "scripts/experiments/run_v24_res1_batch4_lr.py" "${1:?stage required}" "${SLURM_ARRAY_TASK_ID:-0}"
