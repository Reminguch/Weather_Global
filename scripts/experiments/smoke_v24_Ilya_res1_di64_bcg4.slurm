#!/usr/bin/env bash
# Two-update compile/preflight for the new di64/BCG4 legacy and Mamba1 shapes.
#SBATCH --job-name=v24-r1-d64smk
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v24_r1_di64_smoke_%A_%a.out
#SBATCH --error=logs/v24_r1_di64_smoke_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG_DIR="configs/experiments/v24_Ilya/res1_optimizer_matrix_20260904/preflight"
mapfile -t CONFIGS < <(find "${CONFIG_DIR}" -maxdepth 1 -type f -name '*_di64_bcg4_smoke2.json' | sort)
[[ "${#CONFIGS[@]}" -eq 2 ]] || { echo "Expected two preflight configs" >&2; exit 1; }
CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"
RUN_NAME="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"]["run_name"])' "${CONFIG}")"
RUN_DIR="artifacts/checkpoints/v24_Ilya/res1_optimizer_matrix_20260904/preflight/${RUN_NAME}"
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse ${RUN_DIR}" >&2; exit 1; }

export MPLCONFIGDIR="/tmp/mpl-v24-r1-di64-smoke-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "task=${SLURM_ARRAY_TASK_ID} config=${CONFIG} run_dir=${RUN_DIR}"
echo "git_commit=$(git rev-parse HEAD)"
sha256sum "${CONFIG}" src/models/mamba/v24_Ilya/training/endpoint_step.py
python -u scripts/training/train_v24_Ilya.py --config "${CONFIG}"
[[ -f "${RUN_DIR}/checkpoints/checkpoint_step00000002.pkl" ]]
