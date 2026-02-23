#!/bin/bash
# Usage:
#   source scripts/graphcast_env.sh           # activate env
#   source scripts/graphcast_env.sh --setup   # create env (if missing) + install deps + activate

set -eo pipefail

PROJECT_ROOT=/scratch/gpfs/DABANIN/iv9432/Weather_global
ENV_NAME=graphcast311
SETUP_MODE="${1:-}"

cd "${PROJECT_ROOT}"
export PS1="${PS1:-}"

if command -v module >/dev/null 2>&1; then
  module purge
  module load anaconda3/2025.6
fi

if command -v conda >/dev/null 2>&1; then
  NOUNSET_WAS_ON=0
  if [[ $- == *u* ]]; then
    NOUNSET_WAS_ON=1
    set +u
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"

  if [ "${SETUP_MODE}" = "--setup" ]; then
    echo "Setup mode: ensure you are on the LOGIN NODE (compute nodes have no internet)."
    conda create -y -n "${ENV_NAME}" python=3.11
    conda activate "${ENV_NAME}"
    pip install -U pip setuptools wheel
    pip install -r requirements.txt
  else
    conda activate "${ENV_NAME}"
  fi
  if [ "${NOUNSET_WAS_ON}" -eq 1 ]; then
    set -u
  fi
  echo "Using conda env: ${ENV_NAME}"
elif [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
  echo "Using venv: .venv"
else
  echo "No conda/.venv environment found."
  return 1 2>/dev/null || exit 1
fi
