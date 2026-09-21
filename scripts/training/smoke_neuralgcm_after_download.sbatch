#!/usr/bin/env bash
#SBATCH --job-name=ngcm-2p8-smoke
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail
ulimit -c 0
run_root="$1"
data_root="$2"
gpu_python="$3"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
export JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=false
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PYTHONPATH PYTHONHOME
cd "$run_root/code"
/usr/bin/python3 scripts/preprocessing/watch_neuralgcm_smoke.py check --run-root "$run_root"
exec "$gpu_python" -u scripts/training/smoke_neuralgcm_training.py \
  --checkpoint "$run_root/deterministic_2_8_deg.pkl" \
  --prepared "$data_root/res2p8" --output "$run_root/result" --resolution 2.8
