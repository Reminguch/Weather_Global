#!/usr/bin/env bash
# Source this explicitly. Never upgrades the environment used by GC experiments.
NEURALGCM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURALGCM_ENV_PREFIX="${NEURALGCM_ENV_PREFIX:-${NEURALGCM_REPO_ROOT}/.venv_neuralgcm}"
if [[ ! -f "${NEURALGCM_ENV_PREFIX}/bin/activate" ]]; then
  echo "Missing isolated environment: ${NEURALGCM_ENV_PREFIX}" >&2
  return 1 2>/dev/null || exit 1
fi
source "${NEURALGCM_ENV_PREFIX}/bin/activate"
export PYTHONPATH="${NEURALGCM_REPO_ROOT}:${NEURALGCM_REPO_ROOT}/third_party/graphcast:${NEURALGCM_REPO_ROOT}/third_party/neuralgcm${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=false
# The experiment CLI also sets and validates this before importing JAX.
case " ${XLA_FLAGS:-} " in
  *" --xla_gpu_exclude_nondeterministic_ops=true "*) ;;
  *) export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_exclude_nondeterministic_ops=true" ;;
esac
