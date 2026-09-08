#!/usr/bin/env bash
# Exact matched evaluation of SWA(2k,3k,4k,5k,6k,7k) for the ten di16/BCG1
# optimizer-matrix runs that had reached 7k when this comparison was requested.
# Each task uses the same 32 anchors and 40 six-hour leads as the incumbent V22
# legacy SWA(2k--8k) reference.
#SBATCH --job-name=v24-r1-swa27e
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:20:00
#SBATCH --output=logs/v24_r1_swa27_eval_%A_%a.out
#SBATCH --error=logs/v24_r1_swa27_eval_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

CONFIG_DIR="configs/experiments/v24_Ilya/res1_optimizer_matrix_20260904"
EXPERIMENT_ROOT="artifacts/checkpoints/v24_Ilya/res1_optimizer_matrix_20260904"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"
REFERENCE_JSON="artifacts/checkpoints/v22_final/res1_dm_7yr_k20_di_bcg_20k_20260813/di16_bcg1_closed_sg_stateful_20k/eval/cold_full_zero_exact/swa_step02000-08000.json"
ARTIFACT_TAG="swa_step02000-07000"
EVAL_TAG="swa2k7_matched_legacy32"

mapfile -t CONFIGS < <(find "${CONFIG_DIR}" -maxdepth 1 -type f -name '*_di16_bcg1_*.json' | sort)
[[ "${#CONFIGS[@]}" -eq 12 ]] || { echo "Expected 12 di16 configs" >&2; exit 1; }
CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"
RUN_NAME="$(jq -r '.output.run_name' "${CONFIG}")"
RUN_DIR="${EXPERIMENT_ROOT}/${RUN_NAME}"
RUN_CONFIG="${RUN_DIR}/run_config.json"
CKPT="${RUN_DIR}/swa/${ARTIFACT_TAG}.pkl"
OUT_JSON="${RUN_DIR}/eval/cold_full_zero_exact_gc/${EVAL_TAG}/${ARTIFACT_TAG}.json"

for required in "${RUN_CONFIG}" "${CKPT}" "${BASELINE_CKPT}" "${STATS_DIR}" "${EVAL_DATA}" "${REFERENCE_JSON}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${OUT_JSON}" ]] || { echo "Refusing to overwrite ${OUT_JSON}" >&2; exit 1; }
mkdir -p "$(dirname "${OUT_JSON}")"

INIT_SCHEME="$(jq -r '.architecture.temporal_init_scheme' "${RUN_CONFIG}")"
D_INNER="$(jq -r '.architecture.temporal_d_inner' "${RUN_CONFIG}")"
BC_GROUPS="$(jq -r '.architecture.temporal_bc_groups' "${RUN_CONFIG}")"
ZERO_INIT="$(jq -r '.architecture.temporal_zero_init_out' "${RUN_CONFIG}")"
[[ "${ZERO_INIT}" == true ]] || { echo "Expected temporal_zero_init_out=true" >&2; exit 1; }

export MPLCONFIGDIR="/tmp/mpl-v24-r1-swa27-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

echo "run=${RUN_NAME} artifact=${ARTIFACT_TAG} samples=32 output=${OUT_JSON}"
python -u scripts/analyze_models/eval_v24_Ilya.py \
  --ckpt "${CKPT}" --ckpt-in "${BASELINE_CKPT}" \
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
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
  --out-json "${OUT_JSON}"

python - "${OUT_JSON}" "${REFERENCE_JSON}" <<'PY'
import json
import sys
from pathlib import Path

candidate = json.loads(Path(sys.argv[1]).read_text())
reference = json.loads(Path(sys.argv[2]).read_text())
if candidate["chosen_idx"] != reference["chosen_idx"]:
    raise SystemExit("chosen_idx differs from incumbent exact reference")
if candidate["target_steps"] != 40 or candidate["evaluated_samples"] != 32:
    raise SystemExit("evaluation is not complete 32-anchor x 40-lead output")
loss = candidate["original_graphcast_loss"]["improvement_pct_per_step"]
t2m = candidate["per_variable_per_step"]["2m_temperature"]["improvement_pct_rmse"]
print(
    f"day5_allvars_loss_improvement_pct={loss[19]:.6f} "
    f"day10_allvars_loss_improvement_pct={loss[39]:.6f} "
    f"day5_t2m_rmse_improvement_pct={t2m[19]:.6f} "
    f"day10_t2m_rmse_improvement_pct={t2m[39]:.6f}"
)
PY
