#!/bin/bash
#SBATCH --job-name=verify-gpu-jax
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Set these for your cluster if required:
#SBATCH --account=dabanin

set -euo pipefail

PROJECT_ROOT=/scratch/gpfs/DABANIN/iv9432/Weather_global
cd "${PROJECT_ROOT}"

mkdir -p logs

echo "=== SLURM ENV ==="
echo "job_id: ${SLURM_JOB_ID:-unknown}"
echo "node: $(hostname)"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo

echo "=== NVIDIA CHECK ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  echo
  nvidia-smi -L || true
else
  echo "nvidia-smi not found in PATH"
fi
echo

echo "=== PYTHON/JAX CHECK ==="
source scripts/graphcast_env.sh

python -V
python - <<'PY'
import os
import sys

print("python_executable:", sys.executable)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

try:
    import jax
    print("jax_version:", jax.__version__)
    print("jax_backend:", jax.default_backend())
    print("jax_devices:", jax.devices())
except Exception as e:
    print("jax_import_or_runtime_error:", repr(e))
PY
