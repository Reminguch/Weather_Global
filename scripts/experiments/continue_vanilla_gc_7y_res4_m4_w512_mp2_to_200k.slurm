#!/usr/bin/env bash
# Continue the existing seven-year res4/mesh4/width512/mp2 vanilla GraphCast
# run from global step 50,000 to 200,000. The trainer restores parameters but
# initializes a fresh Adam state and restarts the sampling RNG from seed 0.
# Historical evidence: 50k took 0.99 update-hours and peaked near 19.2 GiB RSS.
# The 8h/32G request leaves room for full validation, I/O, and runtime variance.
#SBATCH --job-name=gc-r4-m4-w512-mp2-200k
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/gc_r4_m4_w512_mp2_200k_%j.out
#SBATCH --error=logs/gc_r4_m4_w512_mp2_200k_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-gc-r4-mp2-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"

source scripts/graphcast_env.sh

RUN_PARENT="artifacts/checkpoints/7_years/vanilla_gc"
RUN_NAME="vanilla_gc_7y_res4_m4_w512_mp2_h6_bs8_accum1_stream50k"
RUN_DIR="${RUN_PARENT}/${RUN_NAME}"
CKPT_IN="${RUN_DIR}/ckpt_step50000.npz"
RUN_CONFIG="${RUN_DIR}/run_config.json"

for required in "${RUN_DIR}" "${CKPT_IN}" "${RUN_CONFIG}" \
  data/graphcast/graphcast/dataset/prepared_stream/res4/metadata.json; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required continuation input: ${required}" >&2
    exit 1
  fi
done

if [[ ! -f "${RUN_DIR}/run_config_before_200k_continuation.json" ]]; then
  cp -p "${RUN_CONFIG}" "${RUN_DIR}/run_config_before_200k_continuation.json"
fi

CMD=(
  python -u -m src.models.graphcast.training.standard_training
  --data-path data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr
  --data-source prepared_array
  --prepared-data-root data/graphcast/graphcast/dataset/prepared_stream
  --stats-dir data/graphcast/graphcast/stats
  --ckpt-in "${CKPT_IN}"
  --resume-step 50000
  --out-dir "${RUN_PARENT}"
  --run-name "${RUN_NAME}"
  --val-year 2022
  --train-start-year 2015
  --train-end-year 2021
  --resolution 4.0
  --mesh-size 4
  --width 512
  --processor-msg-steps 2
  --input-duration 12h
  --target-steps 1
  --batch-size 8
  --grad-accum-steps 1
  --eval-batch-size 4
  --eval-num-batches 16
  --final-eval-num-batches all
  --data-cache-mode never
  --data-cache-max-gib 48
  --batch-builder prepared_array
  --prefetch-workers 4
  --prefetch-depth 8
  --prefetch-device-depth 1
  --max-steps 200000
  --eval-every 10000
  --checkpoint-every 10000
  --lr 1e-4
  --weight-decay 1e-4
  --precision bf16
  --seed 0
  --temporal-backbone none
)

echo "Continuing ${RUN_NAME}: global step 50000 -> 200000 (fresh Adam state)."
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Resolved command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
