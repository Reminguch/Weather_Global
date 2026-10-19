#!/usr/bin/env bash
# Identical original reset-inference shards: 22m12s–23m26s, 7.42–7.68 GiB.
#SBATCH --job-name=v24-reset-eval2023
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80&pcie
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:02:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/graphcast_env.sh
WORKFLOW="${1:?workflow required}"
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
export TF_DETERMINISTIC_OPS=0 TF_CUDNN_DETERMINISTIC=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH JAX_PLATFORMS
exec python -u "${WORKFLOW}/control/run_v24_matched_reset_eval.py" evaluate \
  --workflow "${WORKFLOW}" --shard-index "${SLURM_ARRAY_TASK_ID}"
