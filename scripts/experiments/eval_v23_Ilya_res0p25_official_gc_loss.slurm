#!/usr/bin/env bash
# Exact training-objective calibration for the official GraphCast 0.25-degree
# checkpoint. The zero residual head and zero LR make all five updates pure
# frozen-GraphCast evaluations at the K=24 / AR-tail-20 endpoint.
#SBATCH --job-name=v23-r025-gc-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/v23_r025_official_gc_eval_%j.out
#SBATCH --error=logs/v23_r025_official_gc_eval_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG="configs/experiments/v23_Ilya/res0p25_january2021_endpoint_k24_official_gc_eval.json"

export MPLCONFIGDIR="/tmp/mpl-v23-r025-gc-eval-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "config=${CONFIG}"
echo "git_commit=$(git rev-parse HEAD)"
sha256sum "${CONFIG}"

python -u scripts/training/train_v23_Ilya.py --config "${CONFIG}"
