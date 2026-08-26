#!/usr/bin/env bash
# Eight-anchor/40-lead screening for each 1k checkpoint and four SWAs from the
# matched res1 cosine pilot, plus the incumbent SWA(2k--8k). Full 32-anchor
# confirmation is intentionally deferred until this screen selects finalists.
#SBATCH --job-name=v22f-r1-cos-screen
#SBATCH --account=dabanin
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:02:00
#SBATCH --array=0-24
#SBATCH --output=logs/v22f_r1_cos_screen_%A_%a.out
#SBATCH --error=logs/v22f_r1_cos_screen_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"
mkdir -p logs
export MPLCONFIGDIR="/tmp/mpl-v22f-r1-cos-screen-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
source scripts/graphcast_env.sh

RUN_ROOT="artifacts/checkpoints/v22_final/res1_cosine_di16_di64_8k_20260817"
RUN_NAMES=(di16_bcg1_cosine_8k di64_bcg4_cosine_8k)
D_INNERS=(16 64)
BC_GROUPS=(1 4)
SWA_TAGS=(
  swa_step02000-04000
  swa_step02000-06000
  swa_step04000-08000
  swa_step02000-08000
)
REFERENCE_ROOT="artifacts/checkpoints/v22_final/res1_dm_7yr_k20_di_bcg_20k_20260813/di16_bcg1_closed_sg_stateful_20k"
BASELINE_CKPT="data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
STATS_DIR="data/graphcast/graphcast/stats"
EVAL_DATA="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"

task_id="${SLURM_ARRAY_TASK_ID}"
if ((task_id < 24)); then
  variant_id=$((task_id / 12))
  artifact_id=$((task_id % 12))
  run_dir="${RUN_ROOT}/${RUN_NAMES[$variant_id]}"
  d_inner="${D_INNERS[$variant_id]}"
  bc_groups="${BC_GROUPS[$variant_id]}"
  if ((artifact_id < 8)); then
    step=$((1000 * (artifact_id + 1)))
    artifact_tag="step${step}"
    ckpt="${run_dir}/checkpoints/checkpoint_step$(printf '%08d' "${step}").pkl"
  else
    artifact_tag="${SWA_TAGS[$((artifact_id - 8))]}"
    ckpt="${run_dir}/swa/${artifact_tag}.pkl"
  fi
else
  run_dir="${REFERENCE_ROOT}"
  d_inner=16
  bc_groups=1
  artifact_tag="incumbent_swa_step02000-08000"
  ckpt="${run_dir}/swa/swa_step02000-08000.pkl"
fi
out_json="${run_dir}/eval/screen8_cold_full_zero/${artifact_tag}.json"

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
  --temporal-d-inner "${d_inner}" --temporal-d-state 16 \
  --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2 \
  --temporal-bc-groups "${bc_groups}" --temporal-stateful \
  --eval-mode cold_full --residual-state-init zero \
  --residual-alpha 1 --target-steps 40 --warmup-steps 24 \
  --n-samples 8 --seed 0 --out-json "${out_json}"

echo "Completed res1 cosine screen: ${out_json}"
