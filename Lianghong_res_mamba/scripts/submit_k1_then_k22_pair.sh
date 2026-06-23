#!/bin/bash
# Submit two v22-equivalent residual-Mamba curricula:
#   1. current-repo label: K=1 -> K=22
#   2. Lianghong label:   K=1 -> K=22

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

FRESH_SCRIPT=${FRESH_SCRIPT:-Lianghong_res_mamba/scripts/run_v22_fresh.slurm}
RESUME_SCRIPT=${RESUME_SCRIPT:-Lianghong_res_mamba/scripts/run_v22_resume.slurm}

MY_OUT_DIR=${MY_OUT_DIR:-artifacts/checkpoints/res1_official/residual_mamba_v22setup_k1_then_k22}
LIANGHONG_OUT_DIR=${LIANGHONG_OUT_DIR:-artifacts/checkpoints/lianghong_res_mamba}

MY_K1_RUN=${MY_K1_RUN:-my_v22setup_k1}
MY_K22_RUN=${MY_K22_RUN:-my_v22setup_k22_from_k1}
LIANGHONG_K1_RUN=${LIANGHONG_K1_RUN:-lianghong_v22setup_k1}
LIANGHONG_K22_RUN=${LIANGHONG_K22_RUN:-lianghong_v22setup_k22_from_k1}

MAX_STEPS_K1=${MAX_STEPS_K1:-23000}
MAX_STEPS_K22=${MAX_STEPS_K22:-26000}
RESUME_STEP=${RESUME_STEP:-23000}
TARGET_STEPS_K1=${TARGET_STEPS_K1:-1}
TARGET_STEPS_K22=${TARGET_STEPS_K22:-22}

EVAL_EVERY=${EVAL_EVERY:-2000}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-500}
ALLOW_EXISTING=${ALLOW_EXISTING:-0}
DRY_RUN=${DRY_RUN:-0}

DATA_PATH=${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}
DATA_SOURCE=${DATA_SOURCE:-prepared_array}
PREPARED_DATA_ROOT=${PREPARED_DATA_ROOT:-data/graphcast/graphcast/dataset/prepared_stream}
STATS_DIR=${STATS_DIR:-data/graphcast/graphcast/stats}
OFFICIAL_CKPT=${OFFICIAL_CKPT:-"data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"}

RESOLUTION=${RESOLUTION:-1.0}
MESH_SIZE=${MESH_SIZE:-5}
WIDTH=${WIDTH:-512}
MSG=${MSG:-2}
INPUT_DURATION=${INPUT_DURATION:-12h}
VAL_YEAR=${VAL_YEAR:-2022}
TRAIN_START_YEAR=${TRAIN_START_YEAR:-2015}
TRAIN_END_YEAR=${TRAIN_END_YEAR:-2021}
LEN_SEGMENT=${LEN_SEGMENT:-96}
BPTT_STEPS=${BPTT_STEPS:-24}
CHUNK_LOAD_WORKERS=${CHUNK_LOAD_WORKERS:-6}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
EVAL_NUM_SEGMENTS=${EVAL_NUM_SEGMENTS:-16}
FINAL_EVAL_NUM_SEGMENTS=${FINAL_EVAL_NUM_SEGMENTS:-all}
DATA_CACHE_MODE=${DATA_CACHE_MODE:-never}
DATA_CACHE_MAX_GIB=${DATA_CACHE_MAX_GIB:-48}
BATCH_BUILDER=${BATCH_BUILDER:-prepared_array}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
PRECISION=${PRECISION:-bf16}
SEED=${SEED:-22}
MEMORY_MODE=${MEMORY_MODE:-optimal}
AR_LOSS_MODE=${AR_LOSS_MODE:-tail_uniform}
RESIDUAL_AR_FEEDBACK=${RESIDUAL_AR_FEEDBACK:-baseline_plus_residual}
RESIDUAL_OUTPUT_HEAD=${RESIDUAL_OUTPUT_HEAD:-enabled}
TEMPORAL_INSERT_COUNT=${TEMPORAL_INSERT_COUNT:-2}
TEMPORAL_LAYERS=${TEMPORAL_LAYERS:-2}
D_INNER=${D_INNER:-128}
D_STATE=${D_STATE:-16}
D_CONV=${D_CONV:-4}
DT_RANK=${DT_RANK:-auto}

MY_K1_CKPT="${MY_OUT_DIR}/${MY_K1_RUN}/ckpt_step${MAX_STEPS_K1}.npz"
LIANGHONG_K1_CKPT="${LIANGHONG_OUT_DIR}/${LIANGHONG_K1_RUN}/ckpt_step${MAX_STEPS_K1}.npz"

mkdir -p logs

require_path() {
  local kind="$1"
  local path="$2"
  if [[ "${kind}" == "file" && ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 2
  fi
  if [[ "${kind}" == "dir" && ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 2
  fi
}

check_run_dir() {
  local dir="$1"
  if [[ ! -e "${dir}" ]]; then
    return
  fi
  if [[ "${ALLOW_EXISTING}" == "1" ]]; then
    echo "ALLOW_EXISTING=1: target exists and may be reused: ${dir}" >&2
    return
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN warning: target exists: ${dir}" >&2
    return
  fi
  echo "Refusing to submit because target run directory exists: ${dir}" >&2
  echo "Set ALLOW_EXISTING=1 to override." >&2
  exit 2
}

print_cmd() {
  printf '  ' >&2
  printf '%q ' "$@" >&2
  printf '\n' >&2
}

submit_job() {
  local label="$1"
  shift
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] ${label}:" >&2
    print_cmd "$@"
    echo "DRYRUN_${label}"
    return
  fi
  local job_id
  job_id="$("$@")"
  echo "${label}: ${job_id}" >&2
  printf '%s\n' "${job_id}"
}

common_env=(
  "PROJECT_ROOT=${PROJECT_ROOT}"
  "DATA_PATH=${DATA_PATH}"
  "DATA_SOURCE=${DATA_SOURCE}"
  "PREPARED_DATA_ROOT=${PREPARED_DATA_ROOT}"
  "STATS_DIR=${STATS_DIR}"
  "OFFICIAL_CKPT=${OFFICIAL_CKPT}"
  "RESOLUTION=${RESOLUTION}"
  "MESH_SIZE=${MESH_SIZE}"
  "WIDTH=${WIDTH}"
  "MSG=${MSG}"
  "INPUT_DURATION=${INPUT_DURATION}"
  "VAL_YEAR=${VAL_YEAR}"
  "TRAIN_START_YEAR=${TRAIN_START_YEAR}"
  "TRAIN_END_YEAR=${TRAIN_END_YEAR}"
  "LEN_SEGMENT=${LEN_SEGMENT}"
  "BPTT_STEPS=${BPTT_STEPS}"
  "CHUNK_LOAD_WORKERS=${CHUNK_LOAD_WORKERS}"
  "BATCH_SIZE=${BATCH_SIZE}"
  "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  "EVAL_NUM_SEGMENTS=${EVAL_NUM_SEGMENTS}"
  "FINAL_EVAL_NUM_SEGMENTS=${FINAL_EVAL_NUM_SEGMENTS}"
  "EVAL_EVERY=${EVAL_EVERY}"
  "CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
  "DATA_CACHE_MODE=${DATA_CACHE_MODE}"
  "DATA_CACHE_MAX_GIB=${DATA_CACHE_MAX_GIB}"
  "BATCH_BUILDER=${BATCH_BUILDER}"
  "LR=${LR}"
  "WEIGHT_DECAY=${WEIGHT_DECAY}"
  "PRECISION=${PRECISION}"
  "SEED=${SEED}"
  "MEMORY_MODE=${MEMORY_MODE}"
  "AR_LOSS_MODE=${AR_LOSS_MODE}"
  "RESIDUAL_AR_FEEDBACK=${RESIDUAL_AR_FEEDBACK}"
  "RESIDUAL_OUTPUT_HEAD=${RESIDUAL_OUTPUT_HEAD}"
  "TEMPORAL_INSERT_COUNT=${TEMPORAL_INSERT_COUNT}"
  "TEMPORAL_LAYERS=${TEMPORAL_LAYERS}"
  "D_INNER=${D_INNER}"
  "D_STATE=${D_STATE}"
  "D_CONV=${D_CONV}"
  "DT_RANK=${DT_RANK}"
)

require_path file "${FRESH_SCRIPT}"
require_path file "${RESUME_SCRIPT}"
require_path file "${OFFICIAL_CKPT}"
require_path dir "${PREPARED_DATA_ROOT}/res1"
require_path dir "${STATS_DIR}"

check_run_dir "${MY_OUT_DIR}/${MY_K1_RUN}"
check_run_dir "${MY_OUT_DIR}/${MY_K22_RUN}"
check_run_dir "${LIANGHONG_OUT_DIR}/${LIANGHONG_K1_RUN}"
check_run_dir "${LIANGHONG_OUT_DIR}/${LIANGHONG_K22_RUN}"

my_k1_cmd=(
  env "${common_env[@]}"
  "OUT_DIR=${MY_OUT_DIR}"
  "RUN_NAME=${MY_K1_RUN}"
  "TARGET_STEPS=${TARGET_STEPS_K1}"
  "MAX_STEPS=${MAX_STEPS_K1}"
  sbatch --parsable --export=ALL --chdir="${PROJECT_ROOT}" --job-name=my-v22-k1 "${FRESH_SCRIPT}"
)
my_k1_job="$(submit_job my_k1 "${my_k1_cmd[@]}")"

my_k22_cmd=(
  env "${common_env[@]}"
  "OUT_DIR=${MY_OUT_DIR}"
  "RUN_NAME=${MY_K22_RUN}"
  "TARGET_STEPS=${TARGET_STEPS_K22}"
  "MAX_STEPS=${MAX_STEPS_K22}"
  "RESUME_STEP=${RESUME_STEP}"
  "RESUME_CKPT=${MY_K1_CKPT}"
  sbatch --parsable --export=ALL --chdir="${PROJECT_ROOT}" --job-name=my-v22-k22
  --dependency="afterok:${my_k1_job}" "${RESUME_SCRIPT}"
)
my_k22_job="$(submit_job my_k22 "${my_k22_cmd[@]}")"

lianghong_k1_cmd=(
  env "${common_env[@]}"
  "OUT_DIR=${LIANGHONG_OUT_DIR}"
  "RUN_NAME=${LIANGHONG_K1_RUN}"
  "TARGET_STEPS=${TARGET_STEPS_K1}"
  "MAX_STEPS=${MAX_STEPS_K1}"
  sbatch --parsable --export=ALL --chdir="${PROJECT_ROOT}" --job-name=lh-v22-k1 "${FRESH_SCRIPT}"
)
lianghong_k1_job="$(submit_job lianghong_k1 "${lianghong_k1_cmd[@]}")"

lianghong_k22_cmd=(
  env "${common_env[@]}"
  "OUT_DIR=${LIANGHONG_OUT_DIR}"
  "RUN_NAME=${LIANGHONG_K22_RUN}"
  "TARGET_STEPS=${TARGET_STEPS_K22}"
  "MAX_STEPS=${MAX_STEPS_K22}"
  "RESUME_STEP=${RESUME_STEP}"
  "RESUME_CKPT=${LIANGHONG_K1_CKPT}"
  sbatch --parsable --export=ALL --chdir="${PROJECT_ROOT}" --job-name=lh-v22-k22
  --dependency="afterok:${lianghong_k1_job}" "${RESUME_SCRIPT}"
)
lianghong_k22_job="$(submit_job lianghong_k22 "${lianghong_k22_cmd[@]}")"

echo
echo "K=1 -> K=22 pair:"
echo "  my_k1:          ${my_k1_job}"
echo "  my_k22:         ${my_k22_job} (afterok:${my_k1_job})"
echo "  lianghong_k1:   ${lianghong_k1_job}"
echo "  lianghong_k22:  ${lianghong_k22_job} (afterok:${lianghong_k1_job})"

if [[ "${DRY_RUN}" != "1" ]]; then
  submission_log=${SUBMISSION_LOG:-"logs/k1_then_k22_pair_${my_k1_job}_${lianghong_k1_job}.txt"}
  {
    echo "my_k1=${my_k1_job}"
    echo "my_k22=${my_k22_job}"
    echo "lianghong_k1=${lianghong_k1_job}"
    echo "lianghong_k22=${lianghong_k22_job}"
    echo "monitor=squeue -j ${my_k1_job},${my_k22_job},${lianghong_k1_job},${lianghong_k22_job}"
  } > "${submission_log}"
  echo "Recorded submission IDs in ${submission_log}"
  echo "Monitor with:"
  echo "  squeue -j ${my_k1_job},${my_k22_job},${lianghong_k1_job},${lianghong_k22_job}"
fi
