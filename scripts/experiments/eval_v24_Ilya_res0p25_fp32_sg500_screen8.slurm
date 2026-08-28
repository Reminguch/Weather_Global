#!/usr/bin/env bash
# Eight-anchor exact-GC res0.25 screen at every 50-step checkpoint through 500.
#SBATCH --job-name=v24-r025-sg-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/v24_r025_sg_eval_%A_%a.out
#SBATCH --error=logs/v24_r025_sg_eval_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

RUN_DIR="artifacts/checkpoints/v24_Ilya/res0p25_fp32_sg500_20260828/legacy_di16_bcg1_sparse_h20_no_tzero_fp32_lr3em6_sg500"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
EVAL_DATA="data/graphcast/graphcast/dataset/wb2_graphcast37_jan2022_0p25.zarr"
STATS_DIR="data/graphcast/graphcast/stats"
N_SAMPLES="${N_SAMPLES:-8}"

export MPLCONFIGDIR="/tmp/mpl-v24-r025-sg-eval-${SLURM_ARRAY_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

steps=(50 100 150 200 250 300 350 400 450 500)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
if ((task_id < 0 || task_id > 9)); then
  echo "Task ${task_id} is outside 0-9" >&2
  exit 2
fi
step="${steps[${task_id}]}"
artifact_tag="step${step}"
ckpt="${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl"
run_config="${RUN_DIR}/run_config.json"
out_json="${RUN_DIR}/eval/cold_full_zero_exact_gc/screen${N_SAMPLES}/${artifact_tag}.json"

for required in "${ckpt}" "${run_config}" "${BASELINE_CKPT}" "${EVAL_DATA}" "${STATS_DIR}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite existing output: ${out_json}" >&2; exit 1; }
mkdir -p "$(dirname "${out_json}")"

python -u scripts/analyze_models/eval_v24_Ilya.py \
  --ckpt "${ckpt}" --ckpt-in "${BASELINE_CKPT}" \
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
  --n-samples "${N_SAMPLES}" --omit-rms-bias --seed 0 --out-json "${out_json}"

python - "${out_json}" <<'PY'
import json
import sys
from pathlib import Path

metric = json.loads(Path(sys.argv[1]).read_text())["original_graphcast_loss"]
baseline = metric["baseline_per_step"]
full = metric["full_per_step"]
if len(baseline) != 40 or len(full) != 40:
    raise SystemExit("incomplete 40-step exact GraphCast loss")
day10 = 100.0 * (1.0 - full[39] / baseline[39])
through_day10 = 100.0 * (1.0 - sum(full[:40]) / sum(baseline[:40]))
print(
    f"day10_improvement_pct={day10:.6f} "
    f"through_day10_improvement_pct={through_day10:.6f}"
)
PY
