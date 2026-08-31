#!/usr/bin/env bash
# Exact 32-anchor, 10-day evaluations for the live res0.25 uniform-sparse run.
# Each array task evaluates one immutable checkpoint.  Keep checkpoints separate
# instead of sharding anchors: this avoids repeating the expensive XLA compile
# four times per checkpoint while still evaluating several checkpoints in parallel.
# --omit-rms-bias removes only the spatial-bias diagnostic; the original
# GraphCast loss, per-channel errors, and exact vanilla comparison are unchanged.
#SBATCH --job-name=v24-r025-spunint
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v24_r025_spuni_intermediate_eval_%A_%a.out
#SBATCH --error=logs/v24_r025_spuni_intermediate_eval_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

RUN_DIR="artifacts/checkpoints/v24_Ilya/res0p25_sparse_uniform_weight_ablation_20260829/legacy_di16_bcg1_sparse_uniform_h20_no_tzero_fp32_lr3em6_sg500"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
EVAL_DATA="data/graphcast/graphcast/dataset/wb2_graphcast37_jan2022_0p25.zarr"
STATS_DIR="data/graphcast/graphcast/stats"

export MPLCONFIGDIR="/tmp/mpl-v24-r025-spuni-intermediate-${SLURM_ARRAY_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

steps=(50 100 150 200 250)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
if ((task_id < 0 || task_id >= ${#steps[@]})); then
  echo "Task ${task_id} is outside 0-$(( ${#steps[@]} - 1 ))" >&2
  exit 2
fi
step="${steps[${task_id}]}"
checkpoint="${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl"
run_config="${RUN_DIR}/run_config.json"
out_json="${RUN_DIR}/eval/cold_full_zero_exact_gc/reference32/step${step}.json"

for required in "${checkpoint}" "${run_config}" "${BASELINE_CKPT}" "${EVAL_DATA}" "${STATS_DIR}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite existing output: ${out_json}" >&2; exit 1; }
mkdir -p "$(dirname "${out_json}")"

echo "checkpoint=${checkpoint} output=${out_json} anchors=32 leads=40"
python -u scripts/analyze_models/eval_v24_Ilya.py \
  --ckpt "${checkpoint}" --ckpt-in "${BASELINE_CKPT}" \
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
  --n-samples 32 --omit-rms-bias --seed 0 --out-json "${out_json}"

python - "${out_json}" <<'PY'
import json
import sys
from pathlib import Path

metric = json.loads(Path(sys.argv[1]).read_text())["original_graphcast_loss"]
baseline = metric["baseline_per_step"]
full = metric["full_per_step"]
if len(baseline) != 40 or len(full) != 40:
    raise SystemExit("incomplete 40-step exact GraphCast loss")
print(
    f"rollout_improvement_pct={metric['improvement_pct_rollout']:.6f} "
    f"day10_improvement_pct={metric['improvement_pct_per_step'][39]:.6f}"
)
PY
