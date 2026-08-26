#!/usr/bin/env bash
# Confirm, on 32 fixed anchors, the best exact-GC artifact from each of the
# four objective/width runs screened by eval_v23_Ilya_res1_mamba1_loss_di_screen8.slurm.
# Comparable res1 32-anchor evaluations took 32:26--40:37 and 12.1--12.7 GiB.
#SBATCH --job-name=v23-r1-m1-confirm
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:02:00
#SBATCH --array=0-3
#SBATCH --output=logs/v23_r1_m1_confirm_%A_%a.out
#SBATCH --error=logs/v23_r1_m1_confirm_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

EXPERIMENT_ROOT="artifacts/checkpoints/v23_Ilya/res1_mamba1_loss_di_bcg4_8k_20260821"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"

export MPLCONFIGDIR="/tmp/mpl-v23-r1-m1-confirm-${SLURM_ARRAY_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

task_id="${SLURM_ARRAY_TASK_ID:-0}"
case "${task_id}" in
  0)
    run_name="di16_bcg4_sparse_h20_mamba1_closed_sg_stateful_lr1em4_8k"
    d_inner=16
    ;;
  1)
    run_name="di32_bcg4_sparse_h20_mamba1_closed_sg_stateful_lr1em4_8k"
    d_inner=32
    ;;
  2)
    run_name="di16_bcg4_all_h20_mamba1_closed_sg_stateful_lr1em4_8k"
    d_inner=16
    ;;
  3)
    run_name="di32_bcg4_all_h20_mamba1_closed_sg_stateful_lr1em4_8k"
    d_inner=32
    ;;
  *)
    echo "Task ${task_id} is outside 0-3" >&2
    exit 2
    ;;
esac

run_dir="${EXPERIMENT_ROOT}/${run_name}"
screen_dir="${run_dir}/eval/cold_full_zero_exact_gc/screen8"
selection="$({ python - "${screen_dir}" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
paths = sorted(directory.glob("*.json"))
if len(paths) != 6:
    raise RuntimeError(f"Expected six screen JSONs in {directory}, found {len(paths)}")
ranked = []
for path in paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    exact = payload["original_graphcast_loss"]
    if len(exact["baseline_per_step"]) != 40:
        raise RuntimeError(f"Incomplete exact GraphCast metric in {path}")
    ranked.append((float(exact["improvement_pct_rollout"]), path.stem, payload["ckpt"]))
ranked.sort(reverse=True)
_score, artifact, checkpoint = ranked[0]
print(f"{artifact}\t{checkpoint}")
PY
} )"
IFS=$'\t' read -r artifact_tag ckpt <<< "${selection}"

run_config="${run_dir}/run_config.json"
out_json="${run_dir}/eval/cold_full_zero_exact_gc/confirm32/${artifact_tag}.json"
for required in "${ckpt}" "${run_config}" "${BASELINE_CKPT}" "${STATS_DIR}" "${EVAL_DATA}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite existing output: ${out_json}" >&2; exit 1; }
mkdir -p "$(dirname "${out_json}")"

echo "run=${run_name} selected=${artifact_tag} checkpoint=${ckpt} output=${out_json}"
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
  --no-zero-init-out --eval-mode cold_full --residual-state-init zero \
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

echo "Completed 32-anchor exact-GC confirmation: ${out_json}"
