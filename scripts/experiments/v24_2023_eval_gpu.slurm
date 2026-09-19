#!/usr/bin/env bash
# 16 starts: approximately 523s compilation/first + 15*104s steady; 62min allows I/O margin.
# Completed V24 res1 evaluations peaked at 9.5-9.8GiB; allocate 16GiB host RAM.
#SBATCH --job-name=v24-2023-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:02:00
#SBATCH --output=logs/v24_2023_%A_%a.out
#SBATCH --error=logs/v24_2023_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
PHASE="${1:?phase required}"
WORKFLOW="${2:?absolute workflow path required}"
source scripts/graphcast_env.sh
export PYTHONPATH="${WORKFLOW}/source:${WORKFLOW}/source/third_party/graphcast"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_deterministic_ops=true"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/mpl-v24-2023-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
exec python -u "${WORKFLOW}/source/scripts/experiments/run_v24_2023_eval.py" \
  "${PHASE}" --workflow "${WORKFLOW}" --task "${SLURM_ARRAY_TASK_ID}"
