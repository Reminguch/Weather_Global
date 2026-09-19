#!/usr/bin/env bash
# Preparation is CPU-only; the launcher overrides resources for GPU shards.
# Eight-anchor res0.25 shards 13160781_2 / 13160787_1 took 3:16--3:17,
# peaking at 61 GiB host RAM. Use 128G/5h for the larger di64/BCG2 model.
# DP training checkpoints are 3.1G each; preparation retains up to five for SWA.
#SBATCH --job-name=v24-r025-ckpts-prepare
#SBATCH --account=dabanin
#SBATCH --qos=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/v24_r025_posteval_%A_%a.out
#SBATCH --error=logs/v24_r025_posteval_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PHASE="${1:?}"
WORKFLOW="${2:?}"
SNAPSHOT="${WORKFLOW}/source"
if [[ ! -f "${SNAPSHOT}/third_party/graphcast/graphcast/graphcast.py" ]]; then
    echo "Missing vendored GraphCast: include third_party/graphcast in ${SNAPSHOT}" >&2
    exit 1
fi
source scripts/graphcast_env.sh
export PYTHONPATH="${SNAPSHOT}:${SNAPSHOT}/third_party/graphcast${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export MPLCONFIGDIR="/tmp/v24-posteval-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
if [[ "${PHASE}" == evaluate ]]; then
    unset JAX_PLATFORMS
else
    export JAX_PLATFORMS=cpu
fi
exec python -u "${SNAPSHOT}/scripts/experiments/run_v24_post_training_eval.py" "${PHASE}" "${WORKFLOW}"
