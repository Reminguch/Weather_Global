#!/usr/bin/env bash
#SBATCH --job-name=ngcm-k1-physical
#SBATCH --qos=gpu-test
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100&gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/sh4809/weatherforecast/Weather_Global/logs/neuralgcm_physical_eval_20260924/%x-%j.out
#SBATCH --error=/home/sh4809/weatherforecast/Weather_Global/logs/neuralgcm_physical_eval_20260924/%x-%j.err
set -euo pipefail
ulimit -c 0
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/sh4809/weatherforecast/Weather_Global
run_id="$1"
mode="${2:-full}"
extra=()
if [[ "$mode" == "smoke" ]]; then extra+=(--smoke); fi
exec .venv_neuralgcm/bin/python -u scripts/diagnostics/evaluate_neuralgcm_k1_physical.py \
  --plan logs/neuralgcm_physical_eval_20260924/evaluation_plan.json \
  --run-id "$run_id" "${extra[@]}" \
  --output "logs/neuralgcm_physical_eval_20260924/${mode}-${run_id}-${SLURM_JOB_ID}"
