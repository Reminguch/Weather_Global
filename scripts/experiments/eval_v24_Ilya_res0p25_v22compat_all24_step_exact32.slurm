#!/usr/bin/env bash
# Exact 32-anchor evaluation for one atomically completed full-24 checkpoint.
# STEP is supplied by the training wrapper; each checkpoint is independent.
# --omit-rms-bias removes only the memory-heavy spatial-bias diagnostic.
# The canonical original GraphCast loss and vanilla baseline remain exact.
#SBATCH --job-name=v24-r025-f24e
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v24_r025_full24_eval_%j.out
#SBATCH --error=logs/v24_r025_full24_eval_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

STEP="${STEP:?Set STEP to one of 50,100,...,500}"
case " ${STEP} " in
  " 50 "|" 100 "|" 150 "|" 200 "|" 250 "|" 300 "|" 350 "|" 400 "|" 450 "|" 500 ") ;;
  *) echo "STEP must be one of 50,100,...,500; got ${STEP}" >&2; exit 2 ;;
esac

RUN_DIR="${RUN_DIR:-artifacts/checkpoints/v24_Ilya/res0p25_v22compat_all24_fp32_20260901/legacy_di16_bcg1_all24_no_tzero_fp32_lr3em6_sg500}"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
EVAL_DATA="data/graphcast/graphcast/dataset/wb2_graphcast37_jan2022_0p25.zarr"
STATS_DIR="data/graphcast/graphcast/stats"
CHECKPOINT="${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${STEP}").pkl"
RUN_CONFIG="${RUN_DIR}/run_config.json"
OUT_JSON="${RUN_DIR}/eval/cold_full_zero_exact_gc/reference32/step${STEP}.json"

for required in "${CHECKPOINT}" "${RUN_CONFIG}" "${BASELINE_CKPT}" "${EVAL_DATA}" "${STATS_DIR}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUT_JSON}" ]] || { echo "Refusing to overwrite existing output: ${OUT_JSON}" >&2; exit 1; }
mkdir -p "$(dirname "${OUT_JSON}")"

export MPLCONFIGDIR="/tmp/mpl-v24-r025-full24-eval-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "checkpoint=${CHECKPOINT} output=${OUT_JSON} anchors=32 warmup=24 leads=40"
echo "git_commit=$(git rev-parse HEAD)"
sha256sum "${CHECKPOINT}" "${RUN_CONFIG}" \
  src/models/mamba/v24_Ilya/model.py \
  scripts/analyze_models/eval_v24_Ilya.py

python -u scripts/analyze_models/eval_v24_Ilya.py \
  --ckpt "${CHECKPOINT}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 0.25 --mesh-size 6 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2019 --train-end-year 2021 \
  --input-duration 12h --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 16 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-init-scheme legacy_haiku \
  --temporal-dt-init random --temporal-dt-min 0.001 --temporal-dt-max 0.1 \
  --temporal-dt-scale 1.0 --temporal-dt-init-floor 0.0001 \
  --temporal-layers 2 --temporal-bc-groups 1 --temporal-stateful \
  --no-zero-init-out --eval-mode cold_full --residual-state-init zero \
  --residual-alpha 1 --target-steps 40 --warmup-steps 24 \
  --n-samples 32 --omit-rms-bias --seed 0 --out-json "${OUT_JSON}"

python - "${OUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
metric = payload["original_graphcast_loss"]
baseline = metric["baseline_per_step"]
full = metric["full_per_step"]
improvement = metric["improvement_pct_per_step"]
if len(baseline) != 40 or len(full) != 40 or len(improvement) != 40:
    raise SystemExit("incomplete 40-step exact GraphCast loss")
print(
    f"rollout_improvement_pct={metric['improvement_pct_rollout']:.6f} "
    f"day10_step40_improvement_pct={improvement[39]:.6f}"
)
PY
