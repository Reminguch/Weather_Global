#!/usr/bin/env bash
# Compare zero-residual GraphCast and raw step 200 under the exact sparse-H20
# fixed-validation protocol, retaining each supervised horizon loss.
#SBATCH --job-name=v23-r025-val-h
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:02:00
#SBATCH --output=logs/v23_r025_training_val_horizons_%j.out
#SBATCH --error=logs/v23_r025_training_val_horizons_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG="configs/experiments/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp2_2k.json"
CHECKPOINT="artifacts/checkpoints/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp2_hybrid_2k_20260818/di16_bcg1_sparse_h20_closed_sg_stateful_dp2_hybrid_lr3em6_2k/checkpoints/checkpoint_step00000200.pkl"
OUTPUT_ROOT="artifacts/checkpoints/v23_Ilya/calibration_res0p25_sparse_h20"
RUN_NAME="baseline_vs_step200_horizons_${SLURM_JOB_ID}"

export MPLCONFIGDIR="/tmp/mpl-v23-r025-val-h-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "config=${CONFIG}"
echo "checkpoint=${CHECKPOINT}"
echo "output=${OUTPUT_ROOT}/${RUN_NAME}/validation_comparison.json"
echo "git_commit=$(git rev-parse HEAD)"

python -u scripts/training/train_v23_Ilya.py \
  --config "${CONFIG}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --validation-compare "${CHECKPOINT}"
