#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

BRANCH_REF="${BRANCH_REF:-origin/AR-Training-Lianghong}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-/scratch/gpfs/DABANIN/lm8598/Weather_Global/docs/model_source_snapshots/v20_v22_mamba_2026-05-23}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/Lianghong_res_mamba/v22_exact_code/source}"
PROVENANCE_DIR="${PROVENANCE_DIR:-${PROJECT_ROOT}/Lianghong_res_mamba/v22_exact_code/provenance}"

mkdir -p "${SOURCE_ROOT}" "${PROVENANCE_DIR}"

write_from_git() {
  local repo_path="$1"
  local dest="$2"
  mkdir -p "$(dirname "${dest}")"
  git show "${BRANCH_REF}:${repo_path}" > "${dest}"
}

write_tree_from_git() {
  local repo_prefix="$1"
  git ls-tree -r --name-only "${BRANCH_REF}" "${repo_prefix}" | while IFS= read -r repo_path; do
    [[ -n "${repo_path}" ]] || continue
    write_from_git "${repo_path}" "${SOURCE_ROOT}/${repo_path}"
  done
}

copy_snapshot() {
  local snapshot_file="$1"
  local dest="$2"
  mkdir -p "$(dirname "${dest}")"
  cp "${SNAPSHOT_DIR}/${snapshot_file}" "${dest}"
}

echo "[preflight] materializing v22 source under ${SOURCE_ROOT}"
write_from_git "scripts/training/full_mamba_v20/train_mz_v20.py" \
  "${SOURCE_ROOT}/scripts/training/full_mamba_v20/train_mz_v20.py"
write_tree_from_git "src/models/graphcast/training/core"
write_tree_from_git "src/models/mamba/modules"
write_tree_from_git "src/models/mamba/training"
write_tree_from_git "third_party/graphcast/graphcast"

copy_snapshot "temporal_mesh_mamba_Ilya.py" \
  "${SOURCE_ROOT}/src/models/mamba/modules/temporal_mesh_mamba_Ilya.py"
copy_snapshot "runtime.py" \
  "${SOURCE_ROOT}/src/models/mamba/residual_mamba/runtime.py"
copy_snapshot "training_config.py" \
  "${SOURCE_ROOT}/src/models/mamba/residual_mamba/training/config.py"
copy_snapshot "training_model.py" \
  "${SOURCE_ROOT}/src/models/mamba/residual_mamba/training/model.py"
copy_snapshot "training_runner.py" \
  "${SOURCE_ROOT}/src/models/mamba/residual_mamba/training/runner.py"

cat > "${PROVENANCE_DIR}/source_manifest.txt" <<EOF
branch_ref=${BRANCH_REF}
snapshot_dir=${SNAPSHOT_DIR}
source_root=${SOURCE_ROOT}
EOF

find "${SOURCE_ROOT}" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${PROVENANCE_DIR}/source_hashes.sha256"

required_paths=(
  "scripts/training/full_mamba_v20/train_mz_v20.py"
  "scripts/training/full_mamba_v9/train_mz_v9.py"
  "scripts/training/train_graphcast.py"
  "src/data/prepared_array.py"
  "src/models/graphcast/training/core/model.py"
  "src/models/mamba/training/param_utils.py"
  "src/models/mamba/modules/temporal_mesh_mamba_Ilya.py"
  "third_party/graphcast/graphcast/graphcast.py"
)

missing=()
for rel in "${required_paths[@]}"; do
  if [[ ! -f "${SOURCE_ROOT}/${rel}" ]]; then
    missing+=("${rel}")
  fi
done

if (( ${#missing[@]} > 0 )); then
  {
    echo "[preflight] missing required v22 source dependencies:"
    printf '  - %s\n' "${missing[@]}"
    echo
    echo "The harness refuses to substitute active repo code for these files."
    echo "Provide the missing v22 production files under ${SOURCE_ROOT}, then rerun preflight."
  } >&2
  exit 20
fi

echo "[preflight] running v22 trainer import check"
bash -lc "source scripts/graphcast_env.sh && PYTHONPATH='${SOURCE_ROOT}:${SOURCE_ROOT}/third_party/graphcast' python - <<'PY'
import importlib.util
from pathlib import Path

source_root = Path('${SOURCE_ROOT}')
trainer = source_root / 'scripts/training/full_mamba_v20/train_mz_v20.py'
spec = importlib.util.spec_from_file_location('lianghong_v22_train_mz_v20', trainer)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print('import_ok', trainer)
PY"

echo "[preflight] OK"
