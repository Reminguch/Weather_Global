#!/usr/bin/env bash
# Production 0.25° pilot: 32m14s and 26.45 GiB host peak (15634902_0).
# One start, streaming outputs; allow headroom for the upstream scan wrapper.
#SBATCH --job-name=gc025-stock-audit
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:02:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/graphcast_env.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
export TF_DETERMINISTIC_OPS=0 TF_CUDNN_DETERMINISTIC=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH JAX_PLATFORMS
WORKFLOW="${1:?workflow required}"
exec python -u "${WORKFLOW}/audits/stock_source/audit_common_grid_stock.py" --workflow "${WORKFLOW}"
