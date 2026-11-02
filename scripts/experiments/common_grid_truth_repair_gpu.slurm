#!/usr/bin/env bash
# Three initializations per job amortize the production graph compilation.
# Resource overrides and their accounting evidence are recorded in protocol.json.
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
REPAIR_WORKFLOW="${1:?repair workflow required}"
MODEL="${2:?model required}"
REFERENCE_WORKFLOW="${3:?reference workflow required}"
cd "${SLURM_SUBMIT_DIR}"
source scripts/graphcast_env.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
export TF_DETERMINISTIC_OPS=0 TF_CUDNN_DETERMINISTIC=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/mpl-truth-repair-${SLURM_JOB_ID}"
unset PYTHONPATH JAX_PLATFORMS
mkdir -p "${MPLCONFIGDIR}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
exec python -u "${REPAIR_WORKFLOW}/source/scripts/analyze_models/rerun_common_grid_truth_repair.py" \
  --reference-workflow "${REFERENCE_WORKFLOW}" \
  --repair-protocol "${REPAIR_WORKFLOW}/protocol.json" \
  --initializations "${REPAIR_WORKFLOW}/initializations.json" \
  --corrected-truth "${REPAIR_WORKFLOW}/data/common_truth" \
  --output-dir "${REPAIR_WORKFLOW}/shards" \
  --model "${MODEL}" --shard-index 0 --shard-count 1
