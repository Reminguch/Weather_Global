#!/usr/bin/env bash
# Exact GraphCast loss on 40-step res1 cold full-feedback rollouts. Each of
# four ablations contributes step 2k/4k/6k/8k/10k and SWA(2k-8k).
#SBATCH --job-name=v24-r1-v22ev
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --array=0-23
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:02:00
#SBATCH --output=logs/v24_r1_v22ev_%A_%a.out
#SBATCH --error=logs/v24_r1_v22ev_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

EXPERIMENT_ROOT="artifacts/checkpoints/v24_Ilya/res1_v22compat_init_ablation_20260828"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"
N_SAMPLES="${N_SAMPLES:-8}"

export MPLCONFIGDIR="/tmp/mpl-v24-r1-v22ev-${SLURM_ARRAY_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if ((task_id < 0 || task_id > 23)); then
  echo "Task ${task_id} is outside 0-23" >&2
  exit 2
fi
run_index=$((task_id / 6))
artifact_index=$((task_id % 6))

case "${run_index}" in
  0)
    run_name="v22compat_uniform_legacy_di16_bcg1_tzero_fp32_lr1em4_10k"
    init_scheme="legacy_haiku"
    no_tzero=0
    ;;
  1)
    run_name="v22compat_sparse_legacy_di16_bcg1_tzero_fp32_lr1em4_10k"
    init_scheme="legacy_haiku"
    no_tzero=0
    ;;
  2)
    run_name="v22compat_sparse_mamba1_di16_bcg1_tzero_fp32_lr1em4_10k"
    init_scheme="mamba1"
    no_tzero=0
    ;;
  3)
    run_name="v22compat_sparse_mamba1_di16_bcg1_no_tzero_fp32_lr1em4_10k"
    init_scheme="mamba1"
    no_tzero=1
    ;;
esac

run_dir="${EXPERIMENT_ROOT}/${run_name}"
if ((artifact_index < 5)); then
  steps=(2000 4000 6000 8000 10000)
  step="${steps[${artifact_index}]}"
  artifact_tag="step${step}"
  ckpt="${run_dir}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl"
else
  artifact_tag="swa_step02000-08000"
  ckpt="${run_dir}/swa/${artifact_tag}.pkl"
fi

run_config="${run_dir}/run_config.json"
out_json="${run_dir}/eval/cold_full_zero_exact_gc/screen${N_SAMPLES}/${artifact_tag}.json"
for required in "${ckpt}" "${run_config}" "${BASELINE_CKPT}" "${STATS_DIR}" "${EVAL_DATA}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite existing output: ${out_json}" >&2; exit 1; }
mkdir -p "$(dirname "${out_json}")"

zero_flag=()
if ((no_tzero)); then
  zero_flag=(--no-zero-init-out)
fi

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
  --temporal-layers 2 --temporal-bc-groups 1 --temporal-stateful "${zero_flag[@]}" \
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
print(f"day10_improvement_pct={day10:.6f} through_day10_improvement_pct={through_day10:.6f}")
PY
