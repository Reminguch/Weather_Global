#!/usr/bin/env bash
# Patch eval: K=10 & K=14 inner1024 SWAs (were missing when main eval ran).
#SBATCH --job-name=v22cl-res2-Kscan-SWA-eval-patch
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_res2_Kscan_SWA_eval_patch_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_res2_Kscan_SWA_eval_patch_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
DATA=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr

KS=(10 14)
K=${KS[$SLURM_ARRAY_TASK_ID]}
TAG=inner1024
D_INNER=1024

CKPT_DIR=${ROOT}/results/v22_res2_closedloop/K${K}_res2_inner1024
CKPT=${CKPT_DIR}/v13_residual_SWA_step3k-8k.pkl

OUT_DIR=${ROOT}/results/v22_res2_closedloop/Kscan_SWA_eval
mkdir -p ${OUT_DIR}
OUT_JSON=${OUT_DIR}/K${K}_${TAG}_SWA_step3k-8k_cold_full_zero.json

echo "=== patch eval K=${K} ${TAG} ==="
$PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt ${CKPT} --ckpt-in ${GCV1_K3} \
  --data-path ${DATA} \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size ${D_INNER} \
  --temporal-d-inner ${D_INNER} \
  --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --eval-mode cold_full --residual-state-init zero \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
  --out-json "${OUT_JSON}"
echo "=== done K=${K} ${TAG} ==="
