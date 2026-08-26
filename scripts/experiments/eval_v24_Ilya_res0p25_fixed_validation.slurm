#!/usr/bin/env bash
# Matched one-segment training validation for the v24 FP32 zero-residual baseline
# or a native v24 checkpoint. Prior matched jobs used ~110 GiB and ~35 minutes.
#SBATCH --job-name=v24-r025-fixed-val
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:20:00
#SBATCH --output=logs/v24_r025_fixed_validation_%j.out
#SBATCH --error=logs/v24_r025_fixed_validation_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG="configs/experiments/v24_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp2_200.json"
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/checkpoints/v24_Ilya/calibration_res0p25_fp32}"
RUN_NAME="${RUN_NAME:-fixed_validation_${SLURM_JOB_ID}}"

export MPLCONFIGDIR="/tmp/mpl-v24-r025-fixed-val-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

ARGS=(
  --config "${CONFIG}"
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
)
if [[ -n "${CHECKPOINT}" ]]; then
  [[ -f "${CHECKPOINT}" ]] || { echo "Missing native v24 checkpoint: ${CHECKPOINT}" >&2; exit 1; }
  ARGS+=(--validation-compare "${CHECKPOINT}")
  RESULT_NAME="validation_comparison.json"
else
  ARGS+=(--baseline-validation-only)
  RESULT_NAME="baseline_validation.json"
fi

python -u scripts/training/train_v24_Ilya.py "${ARGS[@]}"
python -u scripts/analyze_models/verify_v24_Ilya_fp32_validation.py \
  "${OUTPUT_ROOT}/${RUN_NAME}/${RESULT_NAME}"

