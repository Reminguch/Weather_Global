#!/bin/bash
#SBATCH --job-name=lh-v22-exact
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=340G
#SBATCH --time=48:00:00
#SBATCH --output=logs/lianghong_v22_exact_%j.out
#SBATCH --error=logs/lianghong_v22_exact_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

mkdir -p logs
source scripts/graphcast_env.sh

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/Lianghong_res_mamba/v22_exact_code/source}"
PREPARED_ROOT="${PREPARED_ROOT:-data/graphcast/graphcast/dataset/prepared_stream/res1}"
RESIDUAL_ROOT="${RESIDUAL_ROOT:-data/graphcast/graphcast/dataset/precomputed_residuals/lianghong_v22_iv9432_res1}"
OUT_DIR="${OUT_DIR:-artifacts/checkpoints/lianghong_v22_exact_code_iv9432}"
CKPT_IN="${CKPT_IN:-data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz}"
STATS_DIR="${STATS_DIR:-data/graphcast/graphcast/stats}"

RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
MAX_STEPS="${MAX_STEPS:?MAX_STEPS is required}"
START_STEP="${START_STEP:?START_STEP is required}"
AR_TAIL_K="${AR_TAIL_K:?AR_TAIL_K is required}"
RESUME_FROM="${RESUME_FROM:-}"
DRY_RUN="${DRY_RUN:-0}"

bash Lianghong_res_mamba/v22_exact_code/preflight.sh

if [[ ! -d "${PREPARED_ROOT}" ]]; then
  echo "Missing prepared root: ${PREPARED_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${RESIDUAL_ROOT}" ]]; then
  echo "Missing precomputed residual root: ${RESIDUAL_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${CKPT_IN}" ]]; then
  echo "Missing official checkpoint: ${CKPT_IN}" >&2
  exit 2
fi
if [[ ! -d "${STATS_DIR}" ]]; then
  echo "Missing stats dir: ${STATS_DIR}" >&2
  exit 2
fi
if [[ -n "${RESUME_FROM}" && ! -f "${RESUME_FROM}" ]]; then
  echo "Missing resume checkpoint: ${RESUME_FROM}" >&2
  exit 2
fi

CMD=(
  python -u "${SOURCE_ROOT}/scripts/training/full_mamba_v20/train_mz_v20.py"
  --prepared-root "${PREPARED_ROOT}"
  --residual-root "${RESIDUAL_ROOT}"
  --ckpt-in "${CKPT_IN}"
  --stats-dir "${STATS_DIR}"
  --out-dir "${OUT_DIR}"
  --run-name "${RUN_NAME}"
  --start-step "${START_STEP}"
  --resolution 1.0
  --mesh-size 5
  --width 512
  --baseline-msg-steps 16
  --residual-msg-steps 2
  --input-duration 12h
  --target-steps 1
  --batch-size 1
  --max-steps "${MAX_STEPS}"
  --checkpoint-every 500
  --lr 1e-4
  --weight-decay 1e-4
  --seed 22
  --precision bf16
  --grad-clip 1.0
  --warmup-steps 200
  --sequential-segment-steps 96
  --bptt-steps 24
  --ar-tail-K "${AR_TAIL_K}"
  --temporal-location mesh_processor_interleaved
  --temporal-hidden-size 128
  --temporal-d-state 16
  --temporal-d-conv 4
  --temporal-dt-rank auto
  --temporal-layers 2
)

if [[ -n "${RESUME_FROM}" ]]; then
  CMD+=(--resume-from "${RESUME_FROM}")
fi

printf 'Resolved command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/third_party/graphcast" "${CMD[@]}"

