#!/usr/bin/env bash
# Evaluate the raw 2k endpoint and 0.5k--2k shadow SWA for each res2 pilot LR.
# Matching res2 evaluations take about 29 minutes and 6 GiB; 12G/62m retains
# approximately 2x time and memory headroom.
#SBATCH --job-name=v22f-r2-swa-lr-eval
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:02:00
#SBATCH --array=0-3
#SBATCH --output=logs/v22f_r2_swa_lr_eval_%A_%a.out
#SBATCH --error=logs/v22f_r2_swa_lr_eval_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-v22f-r2-swa-lr-eval-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

RUN_ROOT="artifacts/checkpoints/v22_final/res2_swa_low_lr_continuation_20260816"
LR_TAGS=(lr1em5 lr3em5)
BASELINE_CKPT="artifacts/checkpoints/7_years/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream500k/ckpt_step500000.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"

task_id="${SLURM_ARRAY_TASK_ID}"
lr_id=$((task_id / 2))
artifact_id=$((task_id % 2))
run_dir="${RUN_ROOT}/di16_swa4k12k_${LR_TAGS[$lr_id]}_2k"
if ((artifact_id == 0)); then
  artifact_tag="step2000"
  ckpt="${run_dir}/checkpoints/checkpoint_step00002000.pkl"
else
  artifact_tag="swa_step00500-02000"
  ckpt="${run_dir}/swa/${artifact_tag}.pkl"
fi
out_json="${run_dir}/eval/cold_full_zero/${artifact_tag}.json"

for required in "${ckpt}" "${BASELINE_CKPT}" "${STATS_DIR}" "${EVAL_DATA}"; do
  [[ -e "${required}" ]] || { echo "Missing input: ${required}" >&2; exit 1; }
done
[[ ! -e "${out_json}" ]] || { echo "Refusing to overwrite: ${out_json}" >&2; exit 1; }
mkdir -p "$(dirname "${out_json}")"

python -u scripts/analyze_models/eval_v22_final.py \
  --ckpt "${ckpt}" --ckpt-in "${BASELINE_CKPT}" \
  --data-path "${EVAL_DATA}" --stats-dir "${STATS_DIR}" \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h --temporal-location mesh_processor_interleaved \
  --temporal-d-inner 16 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 --temporal-bc-groups 1 \
  --temporal-stateful --eval-mode cold_full --residual-state-init zero \
  --residual-alpha 1 --target-steps 40 --warmup-steps 24 \
  --n-samples 32 --seed 0 --out-json "${out_json}"

echo "Completed res2 low-LR pilot evaluation: ${out_json}"
