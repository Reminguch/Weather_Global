#!/usr/bin/env bash
# Exact zero-residual GraphCast calibration under the sparse-H20 training
# fixed-validation protocol. No optimizer update is performed.
#SBATCH --job-name=v23-r025-val-gc
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v23_r025_training_val_baseline_%j.out
#SBATCH --error=logs/v23_r025_training_val_baseline_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG="configs/experiments/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp2_2k.json"
OUTPUT_ROOT="artifacts/checkpoints/v23_Ilya/calibration_res0p25_sparse_h20"
RUN_NAME="zero_residual_fixed_validation_${SLURM_JOB_ID}"

export MPLCONFIGDIR="/tmp/mpl-v23-r025-val-gc-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "config=${CONFIG}"
echo "output=${OUTPUT_ROOT}/${RUN_NAME}/baseline_validation.json"
echo "git_commit=$(git rev-parse HEAD)"

python -u scripts/training/train_v23_Ilya.py \
  --config "${CONFIG}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --baseline-validation-only
