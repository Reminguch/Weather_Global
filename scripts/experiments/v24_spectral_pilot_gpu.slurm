#!/usr/bin/env bash
# Eight starts: 523s first + 7*104s steady before spectral/export overhead.
# Comparable V24 res1 paired evaluations: 13740086_6/13617441_3, peak 9.8 GiB.
# 1:02 avoids the one-hour gputest routing boundary and allows pilot I/O overhead.
#SBATCH --job-name=v24-spectral-pilot
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:02:00
#SBATCH --array=0-2
#SBATCH --output=logs/v24_spectral_pilot_%A_%a.out
#SBATCH --error=logs/v24_spectral_pilot_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
WORKFLOW="${1:?absolute workflow path required}"
export PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
source "${WORKFLOW}/source/scripts/graphcast_env.sh"
export PYTHONPATH="${WORKFLOW}/source:${WORKFLOW}/source/third_party/graphcast"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# Match successful reference evaluations; the added deterministic-ops flag
# coincided with ~118s/lead in the cancelled 2022 pilot, outside this budget.
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/mpl-v24-spectral-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
exec python -u "${WORKFLOW}/source/scripts/experiments/run_v24_spectral_pilot.py" \
  evaluate --workflow "${WORKFLOW}" --task "${SLURM_ARRAY_TASK_ID}"
