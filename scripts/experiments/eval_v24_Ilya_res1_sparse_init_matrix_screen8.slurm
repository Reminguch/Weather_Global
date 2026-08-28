#!/usr/bin/env bash
# Eight-anchor exact-GraphCast-loss screen of all planned res1 checkpoints and
# SWA(2k,4k,6k,8k). Tasks 0-5 are legacy BCG1, 6-11 are Mamba1 BCG1,
# and 12-16 are the Mamba1 BCG4 standard recipe.
#SBATCH --job-name=v24-r1-init-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-16
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:02:00
#SBATCH --output=logs/v24_r1_init_eval_%A_%a.out
#SBATCH --error=logs/v24_r1_init_eval_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

EXPERIMENT_ROOT="artifacts/checkpoints/v24_Ilya/res1_fp32_sparse_init_matrix_20260826"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"
N_SAMPLES="${N_SAMPLES:-8}"

export MPLCONFIGDIR="/tmp/mpl-v24-r1-init-eval-${SLURM_ARRAY_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if ((task_id < 0 || task_id > 16)); then
  echo "Task ${task_id} is outside 0-16" >&2
  exit 2
fi

if ((task_id < 6)); then
  run_index=0
  artifact_index="${task_id}"
elif ((task_id < 12)); then
  run_index=1
  artifact_index=$((task_id - 6))
else
  run_index=2
  artifact_index=$((task_id - 12))
fi

case "${run_index}" in
  0)
    run_name="legacy_di16_bcg1_sparse_h20_no_tzero_fp32_lr3em6_10k"
    init_scheme="legacy_haiku"
    bc_groups=1
    final_artifact_index=5
    ;;
  1)
    run_name="mamba1_di16_bcg1_sparse_h20_no_tzero_fp32_lr3em6_10k"
    init_scheme="mamba1"
    bc_groups=1
    final_artifact_index=5
    ;;
  2)
    run_name="mamba1_di16_bcg4_sparse_h20_no_tzero_fp32_lr1em4_8k"
    init_scheme="mamba1"
    bc_groups=4
    final_artifact_index=4
    ;;
esac

run_dir="${EXPERIMENT_ROOT}/${run_name}"
if ((artifact_index < final_artifact_index)); then
  steps=(2000 4000 6000 8000 10000)
  step="${steps[${artifact_index}]}"
  artifact_tag="step${step}"
  ckpt="${run_dir}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl"
else
  artifact_tag="swa_step02000-08000"
  ckpt="${run_dir}/swa/${artifact_tag}.pkl"
  inputs=()
  source_steps=()
  for step in 2000 4000 6000 8000; do
    input="${run_dir}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl"
    [[ -f "${input}" ]] || { echo "Missing SWA input: ${input}" >&2; exit 1; }
    inputs+=("${input}")
    source_steps+=("${step}")
  done
  mkdir -p "$(dirname "${ckpt}")"
  if [[ ! -e "${ckpt}" ]]; then
    python -u scripts/training/build_v24_Ilya_swa.py \
      --inputs "${inputs[@]}" --source-steps "${source_steps[@]}" --output "${ckpt}"
  fi
fi

run_config="${run_dir}/run_config.json"
out_json="${run_dir}/eval/cold_full_zero_exact_gc/screen${N_SAMPLES}/${artifact_tag}.json"
for required in "${ckpt}" "${run_config}" "${BASELINE_CKPT}" "${STATS_DIR}" "${EVAL_DATA}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite existing output: ${out_json}" >&2; exit 1; }
mkdir -p "$(dirname "${out_json}")"

echo "run=${run_name} artifact=${artifact_tag} samples=${N_SAMPLES} output=${out_json}"
python -u scripts/analyze_models/eval_v24_Ilya.py \
  --ckpt "${ckpt}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 16 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-init-scheme "${init_scheme}" \
  --temporal-dt-init random --temporal-dt-min 0.001 --temporal-dt-max 0.1 \
  --temporal-dt-scale 1.0 --temporal-dt-init-floor 0.0001 \
  --temporal-layers 2 --temporal-bc-groups "${bc_groups}" \
  --temporal-stateful --no-zero-init-out \
  --eval-mode cold_full --residual-state-init zero --residual-alpha 1 \
  --target-steps 40 --warmup-steps 24 --n-samples "${N_SAMPLES}" \
  --seed 0 --out-json "${out_json}"

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
