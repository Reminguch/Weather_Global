#!/usr/bin/env bash
# Canonical V22-matched 32-anchor exact evaluation for the full-loss Mamba1
# res1 lane: five retained checkpoints plus SWA(2k--8k).
#SBATCH --job-name=v24-r1-fullmev
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:20:00
#SBATCH --output=logs/v24_r1_full_mamba1_eval_%A_%a.out
#SBATCH --error=logs/v24_r1_full_mamba1_eval_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

RUN_DIR="artifacts/checkpoints/v24_Ilya/res1_full_loss_mamba1_init_ablation_20260901/v22compat_uniform_mamba1_di16_bcg1_tzero_fp32_lr1em4_10k"
EVAL_DIR_TAG="matched_v22_reference32"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"
V22_REFERENCE="artifacts/checkpoints/v22_final/res1_dm_7yr_k20_di_bcg_20k_20260813/di16_bcg1_closed_sg_stateful_20k/eval/cold_full_zero_exact/swa_step02000-08000.json"

export MPLCONFIGDIR="/tmp/mpl-v24-r1-full-mamba1-eval-${SLURM_ARRAY_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if ((task_id < 0 || task_id > 5)); then
  echo "Task ${task_id} is outside 0-5" >&2
  exit 2
fi
if ((task_id < 5)); then
  steps=(2000 4000 6000 8000 10000)
  step="${steps[${task_id}]}"
  artifact_tag="step${step}"
  ckpt="${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl"
else
  artifact_tag="swa_step02000-08000"
  ckpt="${RUN_DIR}/swa/${artifact_tag}.pkl"
fi

out_json="${RUN_DIR}/eval/cold_full_zero_exact_gc/${EVAL_DIR_TAG}/${artifact_tag}.json"
for required in "${ckpt}" "${RUN_DIR}/run_config.json" "${BASELINE_CKPT}" "${STATS_DIR}" "${EVAL_DATA}" "${V22_REFERENCE}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite existing output: ${out_json}" >&2; exit 1; }
mkdir -p "$(dirname "${out_json}")"

echo "artifact=${artifact_tag} samples=32 output=${out_json}"
python -u scripts/analyze_models/eval_v24_Ilya.py \
  --ckpt "${ckpt}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 16 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-init-scheme mamba1 \
  --temporal-dt-init random --temporal-dt-min 0.001 --temporal-dt-max 0.1 \
  --temporal-dt-scale 1.0 --temporal-dt-init-floor 0.0001 \
  --temporal-layers 2 --temporal-bc-groups 1 --temporal-stateful \
  --eval-mode cold_full --residual-state-init zero --residual-alpha 1 \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
  --out-json "${out_json}"

python - "${out_json}" "${V22_REFERENCE}" <<'PY'
import json
import sys
from pathlib import Path

candidate = json.loads(Path(sys.argv[1]).read_text())
reference = json.loads(Path(sys.argv[2]).read_text())
if candidate["chosen_idx"] != reference["chosen_idx"]:
    raise SystemExit("chosen_idx differs from the good V22 exact reference")
metric = candidate["original_graphcast_loss"]
baseline = metric["baseline_per_step"]
full = metric["full_per_step"]
if len(baseline) != 40 or len(full) != 40:
    raise SystemExit("incomplete 40-step exact GraphCast loss")
day10 = 100.0 * (1.0 - full[39] / baseline[39])
through_day10 = 100.0 * (1.0 - sum(full) / sum(baseline))
print(f"day10_improvement_pct={day10:.6f} through_day10_improvement_pct={through_day10:.6f}")
PY
