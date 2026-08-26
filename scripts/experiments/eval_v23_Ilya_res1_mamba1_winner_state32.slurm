#!/usr/bin/env bash
# State-carry diagnostics for the best of the four 32-anchor confirmations.
# Task 0 preserves the 24-step truth-warmed Mamba state; task 1 resets it at
# forecast start. A 1:30 limit accounts for the extra 24 warmup model steps.
#SBATCH --job-name=v23-r1-m1-state
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --array=0-1
#SBATCH --output=logs/v23_r1_m1_state_%A_%a.out
#SBATCH --error=logs/v23_r1_m1_state_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

EXPERIMENT_ROOT="artifacts/checkpoints/v23_Ilya/res1_mamba1_loss_di_bcg4_8k_20260821"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"

export MPLCONFIGDIR="/tmp/mpl-v23-r1-m1-state-${SLURM_ARRAY_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

task_id="${SLURM_ARRAY_TASK_ID:-0}"
case "${task_id}" in
  0)
    eval_mode="warm_full"
    output_mode="warm_full_zero_exact_gc"
    ;;
  1)
    eval_mode="warm_full_reset_state"
    output_mode="warm_full_reset_state_zero_exact_gc"
    ;;
  *)
    echo "Task ${task_id} is outside 0-1" >&2
    exit 2
    ;;
esac

selection="$({ python - "${EXPERIMENT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = sorted(root.glob("*/eval/cold_full_zero_exact_gc/confirm32/*.json"))
if len(paths) != 4:
    raise RuntimeError(f"Expected four confirmation JSONs under {root}, found {len(paths)}")
ranked = []
for path in paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    exact = payload["original_graphcast_loss"]
    ranked.append(
        (
            float(exact["improvement_pct_rollout"]),
            path.stem,
            payload["ckpt"],
            path.parents[3].name,
        )
    )
ranked.sort(reverse=True)
_score, artifact, checkpoint, run_name = ranked[0]
print(f"{artifact}\t{checkpoint}\t{run_name}")
PY
} )"
IFS=$'\t' read -r artifact_tag ckpt run_name <<< "${selection}"
case "${run_name}" in
  di16_*) d_inner=16 ;;
  di32_*) d_inner=32 ;;
  *) echo "Cannot infer d_inner from ${run_name}" >&2; exit 2 ;;
esac

run_dir="${EXPERIMENT_ROOT}/${run_name}"
run_config="${run_dir}/run_config.json"
out_json="${run_dir}/eval/${output_mode}/confirm32/${artifact_tag}.json"
for required in "${ckpt}" "${run_config}" "${BASELINE_CKPT}" "${STATS_DIR}" "${EVAL_DATA}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite existing output: ${out_json}" >&2; exit 1; }
mkdir -p "$(dirname "${out_json}")"

echo "winner=${run_name}/${artifact_tag} mode=${eval_mode} checkpoint=${ckpt} output=${out_json}"
python -u scripts/analyze_models/eval_v23_Ilya.py \
  --ckpt "${ckpt}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h --temporal-location mesh_processor_interleaved \
  --temporal-d-inner "${d_inner}" --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-init-scheme mamba1 \
  --temporal-dt-init random --temporal-dt-min 0.001 --temporal-dt-max 0.1 \
  --temporal-dt-scale 1.0 --temporal-dt-init-floor 0.0001 \
  --temporal-layers 2 --temporal-bc-groups 4 --temporal-stateful \
  --no-zero-init-out --eval-mode "${eval_mode}" --residual-state-init zero \
  --residual-alpha 1 --target-steps 40 --warmup-steps 24 \
  --n-samples 32 --seed 0 --out-json "${out_json}"

python - "${out_json}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metric = json.loads(path.read_text(encoding="utf-8"))["original_graphcast_loss"]
print(
    "exact_gc_rollout_improvement_pct="
    f"{metric['improvement_pct_rollout']:.6f} "
    "day10_improvement_pct="
    f"{metric['improvement_pct_per_step'][39]:.6f}"
)
PY

echo "Completed winner state diagnostic: ${out_json}"
