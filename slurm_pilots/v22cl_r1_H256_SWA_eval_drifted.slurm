#!/usr/bin/env bash
# Eval res=1 H=256 SWA(step 6k-16k) ckpts. 6 K's, cold_full only.
# NOTE: ckpts were trained in the DRIFTED tree (/home/lm8598/Weather_Global_experiments)
# so eval must also run there — v22clean tree loads ckpts with mismatched forward pass.
#SBATCH --job-name=v22cl-r1-H256-SWA-eval-drifted
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --array=0-5
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_r1_H256_SWA_eval_drifted_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_r1_H256_SWA_eval_drifted_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments   # DRIFTED tree — matches training

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
DATA=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr

KS=(4 8 12 16 20 22)
K=${KS[$SLURM_ARRAY_TASK_ID]}

CKPT_DIR=${ROOT}/results/v22closed/K${K}_H256_fresh_20k/v22closed_H256_K${K}_fresh_20k
# Use zero-state SWA variant so drifted eval (which doesn't support --residual-state-init) still zero-inits h
CKPT=${CKPT_DIR}/v13_residual_SWA_step6k-16k_zero_state.pkl
[ -f "${CKPT}" ] || { echo "ERR: missing ${CKPT}"; exit 1; }

OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-07-09-v22cl-r1-H256-SWA-eval
mkdir -p ${OUT_DIR}
OUT_JSON=${OUT_DIR}/v22cl_r1_K${K}_H256_SWA_step6k-16k_cold_full_zero.json

echo "=== v22cl r1 K=${K} H256 SWA cold_full eval (drifted tree) ==="
$PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt ${CKPT} --ckpt-in "${DM_CKPT}" \
  --data-path ${DATA} \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 256 --temporal-d-inner 256 \
  --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --eval-mode cold_full \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
  --out-json "${OUT_JSON}"
echo "Done K=${K}"
