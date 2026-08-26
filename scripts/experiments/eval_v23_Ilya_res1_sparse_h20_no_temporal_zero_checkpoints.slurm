#!/usr/bin/env bash
# Matched 32-anchor/40-lead cold-rollout evaluation of the v23 Res1 sparse-H20
# no-temporal-zero-init run. Tasks 0-4 evaluate the 2k..10k checkpoints;
# task 5 builds and evaluates the requested four-member SWA(2k--8k).
# Comparable Res1 evaluations used 12.3-12.7 GiB and 38-41 min, so 20G/1:02
# retains useful headroom while avoiding the one-hour gputest boundary.
#SBATCH --job-name=v23-r1-noz-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:02:00
#SBATCH --array=0-5
#SBATCH --output=logs/v23_r1_noz_eval_%A_%a.out
#SBATCH --error=logs/v23_r1_noz_eval_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs

RUN_DIR="artifacts/checkpoints/v23_Ilya/res1_sparse_h20_no_temporal_zero_10k_20260817/di16_bcg1_sparse_h20_closed_sg_stateful_no_tzero_lr3em6_10k"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"

export MPLCONFIGDIR="/tmp/mpl-v23-r1-noz-eval-${SLURM_ARRAY_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
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
  step=$((2000 * (task_id + 1)))
  artifact_tag="step${step}"
  ckpt="${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl"
else
  artifact_tag="swa_step02000-08000"
  ckpt="${RUN_DIR}/swa/${artifact_tag}.pkl"
  inputs=()
  source_steps=()
  for step in 2000 4000 6000 8000; do
    input="${RUN_DIR}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl"
    [[ -f "${input}" ]] || { echo "Missing SWA input: ${input}" >&2; exit 1; }
    inputs+=("${input}")
    source_steps+=("${step}")
  done
  [[ ! -e "${ckpt}" ]] || { echo "Refusing to overwrite existing SWA: ${ckpt}" >&2; exit 1; }
  python -u scripts/training/build_v23_Ilya_swa.py \
    --inputs "${inputs[@]}" --source-steps "${source_steps[@]}" --output "${ckpt}"
fi

out_json="${RUN_DIR}/eval/cold_full_zero/${artifact_tag}.json"
for required in "${ckpt}" "${BASELINE_CKPT}" "${STATS_DIR}" "${EVAL_DATA}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite existing output: ${out_json}" >&2; exit 1; }
mkdir -p "$(dirname "${out_json}")"

echo "artifact=${artifact_tag} checkpoint=${ckpt} output=${out_json}"
python -u scripts/analyze_models/eval_v23_Ilya.py \
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

echo "Completed v23 Res1 no-zero-init evaluation: ${out_json}"
