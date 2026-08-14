#!/usr/bin/env bash
# Four-way controlled res1 residual-Mamba ablation:
# d_inner={16,32} x grouped B/C groups={1,4}.
#
# Resource evidence (Della, one A100-80G): a related local seven-year res1 run
# completed 20k updates in about 42 hours. 54 hours adds roughly 29% margin;
# 240 GiB preserves the observed successful res1 CPU-memory allocation.
#SBATCH --job-name=v22-r1-di-bcg
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=2-06:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/v22_r1_di_bcg_%A_%a.out
#SBATCH --error=logs/v22_r1_di_bcg_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIGS=(
  configs/experiments/v22_final/res1_dm_7yr_k20_di16_bcg1_closed_sg_stateful_20k.json
  configs/experiments/v22_final/res1_dm_7yr_k20_di16_bcg4_closed_sg_stateful_20k.json
  configs/experiments/v22_final/res1_dm_7yr_k20_di32_bcg1_closed_sg_stateful_20k.json
  configs/experiments/v22_final/res1_dm_7yr_k20_di32_bcg4_closed_sg_stateful_20k.json
)
CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"

export MPLCONFIGDIR="/tmp/mpl-v22-r1-di-bcg-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "config=${CONFIG}"
echo "git_commit=$(git rev-parse HEAD)"
git status --short
sha256sum \
  "${CONFIG}" \
  src/models/mamba/modules/temporal_mesh_mamba_Ilya.py \
  src/models/mamba/v22_final/config.py \
  src/models/mamba/v22_final/model.py \
  src/models/mamba/v22_final/training/config.py \
  src/models/mamba/v22_final/training/step.py \
  src/models/mamba/v22_final/training/runner.py

python -u scripts/training/train_v22_final.py --config "${CONFIG}"
