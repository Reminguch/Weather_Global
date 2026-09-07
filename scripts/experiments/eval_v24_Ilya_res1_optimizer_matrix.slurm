#!/usr/bin/env bash
# One task per trained run: screen five checkpoints and three SWAs on eight
# anchors, then exact-evaluate the best checkpoint and best SWA on 32 anchors.
# Submit with SHAPE=di16 or SHAPE=di64 and an aftercorr training dependency.
#SBATCH --job-name=v24-r1-optev
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=07:00:00
#SBATCH --output=logs/v24_r1_opt_eval_%A_%a.out
#SBATCH --error=logs/v24_r1_opt_eval_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

SHAPE="${SHAPE:?Submit with --export=ALL,SHAPE=di16 or SHAPE=di64}"
case "${SHAPE}" in
  di16) CONFIG_GLOB="*_di16_bcg1_*.json" ;;
  di64) CONFIG_GLOB="*_di64_bcg4_*.json" ;;
  *) echo "Unsupported SHAPE=${SHAPE}; expected di16 or di64" >&2; exit 2 ;;
esac

CONFIG_DIR="configs/experiments/v24_Ilya/res1_optimizer_matrix_20260904"
EXPERIMENT_ROOT="artifacts/checkpoints/v24_Ilya/res1_optimizer_matrix_20260904"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"
V22_REFERENCE="artifacts/checkpoints/v22_final/res1_dm_7yr_k20_di_bcg_20k_20260813/di16_bcg1_closed_sg_stateful_20k/eval/cold_full_zero_exact/swa_step02000-08000.json"

mapfile -t CONFIGS < <(find "${CONFIG_DIR}" -maxdepth 1 -type f -name "${CONFIG_GLOB}" | sort)
[[ "${#CONFIGS[@]}" -eq 12 ]] || { echo "Expected 12 configs" >&2; exit 1; }
CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"
RUN_NAME="$(jq -r '.output.run_name' "${CONFIG}")"
RUN_DIR="${EXPERIMENT_ROOT}/${RUN_NAME}"
RUN_CONFIG="${RUN_DIR}/run_config.json"

for required in "${CONFIG}" "${RUN_CONFIG}" "${BASELINE_CKPT}" "${STATS_DIR}" "${EVAL_DATA}" "${V22_REFERENCE}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done

INIT_SCHEME="$(jq -r '.architecture.temporal_init_scheme' "${RUN_CONFIG}")"
D_INNER="$(jq -r '.architecture.temporal_d_inner' "${RUN_CONFIG}")"
BC_GROUPS="$(jq -r '.architecture.temporal_bc_groups' "${RUN_CONFIG}")"
ZERO_INIT="$(jq -r '.architecture.temporal_zero_init_out' "${RUN_CONFIG}")"
[[ "${ZERO_INIT}" == true ]] || { echo "Expected temporal_zero_init_out=true" >&2; exit 1; }

export MPLCONFIGDIR="/tmp/mpl-v24-r1-opt-eval-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

artifact_checkpoint() {
  local tag="$1"
  case "${tag}" in
    step[0-9]*)
      local step="${tag#step}"
      printf '%s/checkpoints/checkpoint_step%08d.pkl\n' "${RUN_DIR}" "${step}"
      ;;
    swa_step*) printf '%s/swa/%s.pkl\n' "${RUN_DIR}" "${tag}" ;;
    *) echo "Unsupported artifact: ${tag}" >&2; return 2 ;;
  esac
}

evaluate_artifact() {
  local tag="$1"
  local samples="$2"
  local eval_dir="$3"
  local ckpt out_json
  ckpt="$(artifact_checkpoint "${tag}")"
  out_json="${RUN_DIR}/eval/cold_full_zero_exact_gc/${eval_dir}/${tag}.json"
  [[ -f "${ckpt}" ]] || { echo "Missing checkpoint: ${ckpt}" >&2; return 1; }
  [[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite ${out_json}" >&2; return 1; }
  mkdir -p "$(dirname "${out_json}")"

  echo "run=${RUN_NAME} artifact=${tag} samples=${samples} output=${out_json}"
  python -u scripts/analyze_models/eval_v24_Ilya.py \
    --ckpt "${ckpt}" --ckpt-in "${BASELINE_CKPT}" \
    --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
    --resolution 1.0 --mesh-size 5 --width 512 \
    --baseline-msg-steps 16 --residual-msg-steps 2 \
    --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
    --input-duration 12h --temporal-location mesh_processor_interleaved \
    --temporal-d-inner "${D_INNER}" --temporal-d-state 16 --temporal-d-conv 4 \
    --temporal-dt-rank auto --temporal-init-scheme "${INIT_SCHEME}" \
    --temporal-dt-init random --temporal-dt-min 0.001 --temporal-dt-max 0.1 \
    --temporal-dt-scale 1.0 --temporal-dt-init-floor 0.0001 \
    --temporal-layers 2 --temporal-bc-groups "${BC_GROUPS}" --temporal-stateful \
    --eval-mode cold_full --residual-state-init zero --residual-alpha 1 \
    --target-steps 40 --warmup-steps 24 --n-samples "${samples}" \
    --seed 0 --out-json "${out_json}"

  python - "${out_json}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
metric = payload["original_graphcast_loss"]
if len(metric["baseline_per_step"]) != 40 or len(metric["full_per_step"]) != 40:
    raise SystemExit("incomplete 40-step exact GraphCast loss")
print(f"rollout_improvement_pct={metric['improvement_pct_rollout']:.6f}")
PY
}

ARTIFACTS=(
  step2000 step4000 step6000 step8000 step10000
  swa_step02000-08000 swa_step04000-08000 swa_step06000-10000
)
for artifact in "${ARTIFACTS[@]}"; do
  evaluate_artifact "${artifact}" 8 screen8
done

mapfile -t SELECTED < <(python - "${RUN_DIR}/eval/cold_full_zero_exact_gc/screen8" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in root.glob("*.json"):
    payload = json.loads(path.read_text())
    rows.append((payload["original_graphcast_loss"]["improvement_pct_rollout"], path.stem))
for prefix in ("step", "swa_"):
    candidates = [row for row in rows if row[1].startswith(prefix)]
    if not candidates:
        raise SystemExit(f"no screen results for {prefix}")
    print(max(candidates)[1])
PY
)
[[ "${#SELECTED[@]}" -eq 2 ]] || { echo "Expected checkpoint and SWA selections" >&2; exit 1; }

printf 'selected checkpoint=%s swa=%s\n' "${SELECTED[0]}" "${SELECTED[1]}"
for artifact in "${SELECTED[@]}"; do
  evaluate_artifact "${artifact}" 32 matched_v22_reference32
  python - "${RUN_DIR}/eval/cold_full_zero_exact_gc/matched_v22_reference32/${artifact}.json" "${V22_REFERENCE}" <<'PY'
import json
import sys
from pathlib import Path

candidate = json.loads(Path(sys.argv[1]).read_text())
reference = json.loads(Path(sys.argv[2]).read_text())
if candidate["chosen_idx"] != reference["chosen_idx"]:
    raise SystemExit("chosen_idx differs from the V22 exact reference")
PY
done
