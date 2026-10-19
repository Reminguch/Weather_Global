#!/usr/bin/env bash
# Pilot-based resource overrides are recorded by run_common_grid_2023.py.
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:02:00
set -euo pipefail
WORKFLOW="${1:?workflow required}"
MODEL="${2:?model required}"
SHARDS="${3:?shard count required}"
OUTPUT="${4:?output directory required}"
shift 4
cd "${SLURM_SUBMIT_DIR}"
source scripts/graphcast_env.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
export TF_DETERMINISTIC_OPS=0 TF_CUDNN_DETERMINISTIC=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="${WORKFLOW}/source:${WORKFLOW}/source/third_party/graphcast"
export MPLCONFIGDIR="/tmp/mpl-common-grid-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
exec python -u "${WORKFLOW}/source/scripts/analyze_models/eval_common_grid_2023.py" \
  --workflow "${WORKFLOW}" --model "${MODEL}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count "${SHARDS}" --output-dir "${OUTPUT}" "$@"
