#!/usr/bin/env bash
# Native Figure 2a pilot 15868012_0: four starts, 5m47s, 15.97 GiB host peak.
# Annual shards contain <=37 starts, about 32s each after compilation.
# Twenty shards fit gpu-test's 25-submission cap alongside the five pilot jobs.
#SBATCH --job-name=gc025-paper-fig2a
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:35:00
set -euo pipefail
cd /scratch/gpfs/DABANIN/iv9432/Weather_global
source scripts/graphcast_env.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
export TF_DETERMINISTIC_OPS=0 TF_CUDNN_DETERMINISTIC=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH JAX_PLATFORMS
WORKFLOW="${1:?workflow required}"
INDEX="${SLURM_ARRAY_TASK_ID:-${2:-0}}"
exec python -u "${WORKFLOW}/eval_source/scripts/analyze_models/eval_graphcast_fig2a.py" \
  --workflow "${WORKFLOW}" --shard-index "${INDEX}"
