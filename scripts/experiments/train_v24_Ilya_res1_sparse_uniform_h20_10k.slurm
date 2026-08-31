#!/usr/bin/env bash
# Isolated res1 sparse-H20 weight ablation.  The sole scientific change from
# the active sparse legacy lane is uniform coefficients on all six horizons.
# The final residual head and internal temporal output remain zero-initialized.
#SBATCH --job-name=v24-r1-spuni
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/v24_r1_spuni_%j.out
#SBATCH --error=logs/v24_r1_spuni_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG="configs/experiments/v24_Ilya/res1_v22compat_sparse_uniform_legacy_di16_bcg1_fp32_10k.json"
OUTPUT_ROOT="artifacts/checkpoints/v24_Ilya/res1_sparse_uniform_weight_ablation_20260828"
RUN_NAME="v22compat_sparse_uniform_h20_legacy_di16_bcg1_tzero_fp32_lr1em4_10k"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
GPU_TELEMETRY="logs/v24_r1_spuni_${SLURM_JOB_ID}_gpu.csv"

[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 1; }
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse existing run: ${RUN_DIR}" >&2; exit 1; }

export MPLCONFIGDIR="/tmp/mpl-v24-r1-spuni-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "config=${CONFIG} run_dir=${RUN_DIR}"
echo "git_commit=$(git rev-parse HEAD)"
git status --short
sha256sum "${CONFIG}" \
  src/models/mamba/v24_Ilya/model.py \
  src/models/mamba/v24_Ilya/training/data.py \
  src/models/mamba/v24_Ilya/training/endpoint_step.py \
  src/models/mamba/v24_Ilya/training/runner.py

monitor_gpu() {
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    sleep 10
  done
}
monitor_gpu > "${GPU_TELEMETRY}" &
GPU_MONITOR_PID=$!
cleanup() {
  kill "${GPU_MONITOR_PID}" 2>/dev/null || true
  wait "${GPU_MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python -u scripts/training/train_v24_Ilya.py --config "${CONFIG}"

python - "${CONFIG}" "${RUN_DIR}/run_config.json" <<'PY'
import json
import sys
from pathlib import Path

from src.models.mamba.v24_Ilya.training.config import load_training_config

candidate_path = Path(sys.argv[1])
saved_path = Path(sys.argv[2])
heavy_tail_path = Path(
    "configs/experiments/v24_Ilya/"
    "res1_v22compat_sparse_legacy_di16_bcg1_fp32_10k.json"
)
candidate = load_training_config(candidate_path).to_dict()
saved = json.loads(saved_path.read_text())
saved.pop("config_source")
saved.pop("derived")
if saved != candidate:
    raise SystemExit("saved run config differs from the resolved candidate config")
heavy_tail = load_training_config(heavy_tail_path).to_dict()
for payload in (candidate, heavy_tail):
    payload.pop("objective")
    payload.pop("output")
if candidate != heavy_tail:
    raise SystemExit("uniform-sparse run changed non-objective settings")
print("verified saved config and sparse-weight-only isolation")
PY

for step in 2000 4000 6000 8000 10000; do
  [[ -f "${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl" ]]
done

python -u scripts/training/build_v24_Ilya_swa.py \
  --inputs \
    "${RUN_DIR}/checkpoints/checkpoint_step00002000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00004000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00006000.pkl" \
    "${RUN_DIR}/checkpoints/checkpoint_step00008000.pkl" \
  --source-steps 2000 4000 6000 8000 \
  --output "${RUN_DIR}/swa/swa_step02000-08000.pkl"

GPU_PEAK_MIB="$(awk -F',' 'BEGIN { peak=0 } { gsub(/ /, "", $4); if (($4 + 0) > peak) peak=$4 + 0 } END { print peak }' "${GPU_TELEMETRY}")"
CPU_MAX_RSS="$(sstat -j "${SLURM_JOB_ID}.batch" --noheader --format=MaxRSS -P 2>/dev/null | tail -n 1 || true)"
echo "resource_summary sampled_gpu_peak_mib=${GPU_PEAK_MIB} cpu_max_rss=${CPU_MAX_RSS}"
