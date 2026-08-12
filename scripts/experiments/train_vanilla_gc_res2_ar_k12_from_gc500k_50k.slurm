#!/usr/bin/env bash
# Continue the res2 GraphCast architecture from the 500k vanilla checkpoint
# with a true 12-step (72-hour) autoregressive target loss.  No Mamba module
# is inserted; all GraphCast parameters remain trainable.
#
# --resume-step 0 is intentional: standard_training loads ckpt-in parameters
# when resume_step is set, while the new output directory stays independent of
# the source run and records a fresh 50k-update experiment.
#
# The source 500k run peaked at 68.0 GiB CPU RSS.  A 12-step target increases
# the live trajectory, so request 96G and the user-requested 24-hour walltime.
#SBATCH --job-name=gc-r2-gc500k-ar-k12
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gc_r2_gc500k_ar_k12_%j.out
#SBATCH --error=logs/gc_r2_gc500k_ar_k12_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT}"

mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-weather-global-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

source scripts/graphcast_env.sh

BASELINE_CKPT="artifacts/checkpoints/7_years/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream500k/ckpt_step500000.npz"
OUT_DIR="${OUT_DIR:-artifacts/checkpoints/7_years/vanilla_gc_res2_ar_k12_from_gc500k_50k}"
RUN_NAME="${RUN_NAME:-vanilla_gc_res2_m4_w512_mp6_ar_k12_from_gc500k_50000steps}"
MAX_STEPS="${MAX_STEPS:-50000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"
EVAL_EVERY="${EVAL_EVERY:-2000}"

[[ -f "${BASELINE_CKPT}" ]] || { echo "Missing baseline: ${BASELINE_CKPT}" >&2; exit 1; }

CMD=(
  python -u -m src.models.graphcast.training.standard_training
  --data-path data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr
  --data-source prepared_array
  --prepared-data-root data/graphcast/graphcast/dataset/prepared_stream
  --stats-dir data/graphcast/graphcast/stats
  --ckpt-in "${BASELINE_CKPT}" --resume-step 0
  --out-dir "${OUT_DIR}" --run-name "${RUN_NAME}"
  --val-year 2022 --train-start-year 2015 --train-end-year 2021
  --resolution 2.0 --mesh-size 4 --width 512 --processor-msg-steps 6
  --input-duration 12h --target-steps 12
  --batch-size 8 --grad-accum-steps 1 --eval-batch-size 4
  --eval-num-batches 16 --final-eval-num-batches all
  --data-cache-mode never --data-cache-max-gib 48 --batch-builder prepared_array
  --prefetch-workers 4 --prefetch-depth 8 --prefetch-device-depth 1
  --max-steps "${MAX_STEPS}" --eval-every "${EVAL_EVERY}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --lr 1e-4 --weight-decay 1e-4 --precision bf16 --seed 0
  --temporal-backbone none
)

echo "GraphCast AR k=12: source=${BASELINE_CKPT} run=${RUN_NAME} max_steps=${MAX_STEPS}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Resolved command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi
"${CMD[@]}"
