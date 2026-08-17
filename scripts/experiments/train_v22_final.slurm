#!/usr/bin/env bash
# Generic v22_final trainer. Set CONFIG to one canonical experiment JSON.
# Completed v20 array 11317907 used about 24 GiB and 8:45 per 50k task;
# request 32 GiB and 24 hours for the maintained parity implementation.
#SBATCH --job-name=train-v22-final
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_v22_final_%j.out
#SBATCH --error=logs/train_v22_final_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-weather-global-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

if [[ -z "${CONFIG:-}" ]]; then
  echo "Set CONFIG to a v22_final training JSON." >&2
  exit 2
fi

CMD=(python -u scripts/training/train_v22_final.py --config "${CONFIG}")
[[ -n "${RESUME:-}" ]] && CMD+=(--resume "${RESUME}")
[[ -n "${INIT_FROM:-}" ]] && CMD+=(--init-from "${INIT_FROM}")
[[ -n "${MAX_STEPS:-}" ]] && CMD+=(--max-steps "${MAX_STEPS}")
[[ -n "${CHECKPOINT_EVERY:-}" ]] && CMD+=(--checkpoint-every "${CHECKPOINT_EVERY}")
[[ -n "${LEARNING_RATE:-}" ]] && CMD+=(--learning-rate "${LEARNING_RATE}")
[[ -n "${OUTPUT_ROOT:-}" ]] && CMD+=(--output-root "${OUTPUT_ROOT}")
[[ -n "${RUN_NAME:-}" ]] && CMD+=(--run-name "${RUN_NAME}")
[[ "${ZERO_STATE_ON_INIT:-0}" == "1" ]] && CMD+=(--zero-state-on-init)
[[ "${DRY_RUN:-0}" == "1" ]] && CMD+=(--dry-run)

printf 'Resolved command:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
