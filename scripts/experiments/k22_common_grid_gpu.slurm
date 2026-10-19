#!/usr/bin/env bash
# Matched historical K22 pilot used 10.87 GiB host RAM and 8m31s (15750888).
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
set -eo pipefail
WORKFLOW="${1:?workflow required}"
COUNT="${2:?shard count required}"
OUTPUT="${3:?output directory required}"
shift 3
cd "${SLURM_SUBMIT_DIR}"
source scripts/graphcast_env.sh
export PYTHONPATH="${WORKFLOW}/source:${WORKFLOW}/source/third_party/graphcast"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
export TF_DETERMINISTIC_OPS=0 TF_CUDNN_DETERMINISTIC=0
exec python -u "${WORKFLOW}/source/scripts/analyze_models/eval_k22_common_grid_2022.py" \
  --workflow "${WORKFLOW}" --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --shard-count "${COUNT}" --output-dir "${OUTPUT}" "$@"
