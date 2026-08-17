#!/usr/bin/env bash
# Evaluate the six unevaluated early continuation checkpoints plus the original
# SWA(2k--8k) reference. Each task uses the same 32 cold anchors and 40 leads.
# Prior matching tasks took 40:01--40:31 and 12.4--12.6 GiB. Use 1:02 rather
# than 1:00 so Della does not route these evaluations to the constrained
# gputest partition.
#SBATCH --job-name=v22f-r1-swa-early-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:02:00
#SBATCH --array=0-6
#SBATCH --output=logs/v22f_r1_swa_early_eval_%A_%a.out
#SBATCH --error=logs/v22f_r1_swa_early_eval_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-v22f-r1-swa-early-eval-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

RUN_ROOT="artifacts/checkpoints/v22_final/res1_swa_low_lr_continuation_20260816"
REFERENCE_ROOT="artifacts/checkpoints/v22_final/res1_dm_7yr_k20_di_bcg_20k_20260813/di16_bcg1_closed_sg_stateful_20k"
LR_TAGS=(lr1em5 lr3em5)
STEPS=(500 1000 1500)
STEP_TAGS=(00000500 00001000 00001500)
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"

task_id="${SLURM_ARRAY_TASK_ID}"
if ((task_id < 6)); then
  lr_id=$((task_id / 3))
  step_id=$((task_id % 3))
  run_dir="${RUN_ROOT}/di16_bcg1_swa2k8k_${LR_TAGS[$lr_id]}_2k"
  artifact_tag="step${STEPS[$step_id]}"
  ckpt="${run_dir}/checkpoints/checkpoint_step${STEP_TAGS[$step_id]}.pkl"
  out_json="${run_dir}/eval/cold_full_zero/${artifact_tag}.json"
else
  run_dir="${REFERENCE_ROOT}"
  artifact_tag="swa_step02000-08000"
  ckpt="${run_dir}/swa/${artifact_tag}.pkl"
  # The older JSON lacks exact original_graphcast_loss, so retain it and write
  # the maintained-evaluator result separately.
  out_json="${run_dir}/eval/cold_full_zero_exact/${artifact_tag}.json"
fi

for required in "${ckpt}" "${BASELINE_CKPT}" "${STATS_DIR}" "${EVAL_DATA}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite: ${out_json}" >&2; exit 1; }
mkdir -p "$(dirname "${out_json}")"

python -u scripts/analyze_models/eval_v22_final.py \
  --ckpt "${ckpt}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 16 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 --temporal-bc-groups 1 \
  --temporal-stateful --eval-mode cold_full --residual-state-init zero \
  --residual-alpha 1 --target-steps 40 --warmup-steps 24 \
  --n-samples 32 --seed 0 --out-json "${out_json}"

echo "Completed res1 early-checkpoint evaluation: ${out_json}"
