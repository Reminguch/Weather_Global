#!/usr/bin/env bash
# One GPU runs both numerical modes so hardware cannot confound the comparison.
set -euo pipefail
EXPERIMENT="${1:?experiment}"
OUTPUT="${2:?new diagnostic directory}"
WORKSPACE="${3:?workspace}"
SOURCE_ROOT="${EXPERIMENT}/source"
export PROJECT_ROOT="${WORKSPACE}"
source "${SOURCE_ROOT}/scripts/graphcast_env.sh"
export PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/third_party/graphcast"
export PYTHONSAFEPATH=1
export WEATHER_WORKSPACE="${WORKSPACE}" WEATHER_SOURCE_ROOT="${SOURCE_ROOT}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
unset JAX_PLATFORMS
DIAGNOSTIC="${OUTPUT}/diagnose_cached_k1_reproducibility.py"
nvidia-smi --query-gpu=uuid,name,driver_version --format=csv > "${OUTPUT}/gpu.csv"
failed=0
for mode in ordinary deterministic; do
    export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
    if [[ "${mode}" == deterministic ]]; then
        export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops=true"
    fi
    for phase in one-step rollout; do
        if ! python -u "${DIAGNOSTIC}" "${phase}" --experiment "${EXPERIMENT}" --output "${OUTPUT}" --mode "${mode}"; then
            failed=1
            echo "Diagnostic phase failed: ${mode}/${phase}" >&2
        fi
        python "${DIAGNOSTIC}" summarize --experiment "${EXPERIMENT}" --output "${OUTPUT}"
    done
done
exit "${failed}"
