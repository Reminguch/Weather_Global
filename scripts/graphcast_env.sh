#!/bin/bash
# Usage:
#   source scripts/graphcast_env.sh           # activate env
#   source scripts/graphcast_env.sh --setup   # create env (if missing) + install deps + activate

set -eo pipefail

PROJECT_ROOT=/scratch/gpfs/MENGDIW/sh4809/Weather_Global
ENV_NAME=graphcast311
SETUP_MODE="${1:-}"
CONDA_ROOT="${PROJECT_ROOT}/.conda"
CONDA_ENVS_PATH="${CONDA_ROOT}/envs"
CONDA_PKGS_DIRS="${CONDA_ROOT}/pkgs"
PIP_CACHE_DIR="${PROJECT_ROOT}/.cache/pip"
TMPDIR="${PROJECT_ROOT}/.tmp"
ENV_PREFIX="${CONDA_ENVS_PATH}/${ENV_NAME}"

cd "${PROJECT_ROOT}"
export PS1="${PS1:-}"
mkdir -p "${CONDA_ENVS_PATH}" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${TMPDIR}"
export CONDA_ENVS_PATH CONDA_PKGS_DIRS PIP_CACHE_DIR TMPDIR XDG_CACHE_HOME="${PROJECT_ROOT}/.cache"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  if ! module load anaconda3/2025.6 >/dev/null 2>&1; then
    echo "Warning: module load anaconda3/2025.6 failed; trying direct conda init." >&2
  fi
fi

if [ -f "/usr/licensed/anaconda3/2025.6/etc/profile.d/conda.sh" ]; then
  NOUNSET_WAS_ON=0
  if [[ $- == *u* ]]; then
    NOUNSET_WAS_ON=1
    set +u
  fi
  # shellcheck source=/dev/null
  source "/usr/licensed/anaconda3/2025.6/etc/profile.d/conda.sh"
  if [ "${NOUNSET_WAS_ON}" -eq 1 ]; then
    set -u
  fi
fi

if command -v conda >/dev/null 2>&1; then
  NOUNSET_WAS_ON=0
  if [[ $- == *u* ]]; then
    NOUNSET_WAS_ON=1
    set +u
  fi

  if [ "${SETUP_MODE}" = "--setup" ]; then
    echo "Setup mode: ensure you are on the LOGIN NODE (compute nodes have no internet)."
    conda create -y -p "${ENV_PREFIX}" python=3.11
    conda activate "${ENV_PREFIX}"
    pip install -U pip setuptools wheel
    pip install -r requirements.txt
  else
    if [ -d "${ENV_PREFIX}" ]; then
      conda activate "${ENV_PREFIX}"
    else
      conda activate "${ENV_NAME}"
    fi
  fi
  if [ "${NOUNSET_WAS_ON}" -eq 1 ]; then
    set -u
  fi
  echo "Using conda env: ${CONDA_PREFIX}"
elif [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
  echo "Using venv: .venv"
else
  echo "No conda/.venv environment found."
  return 1 2>/dev/null || exit 1
fi
