#!/usr/bin/env bash
# Eval SWA(step 3k-8k) for res=2 K-scan: 2 configs × 10 K = 20 tasks.
# Configs: inner128, inner1024. K ∈ {2,4,6,8,10,12,14,16,20,22} (K=18 done in arch-screen).
#SBATCH --job-name=v22cl-res2-Kscan-SWA-eval
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --array=0-19
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_res2_Kscan_SWA_eval_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_res2_Kscan_SWA_eval_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
DATA=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr

KS=(2 4 6 8 10 12 14 16 20 22)
# Array idx encoding: 0..9 = inner128; 10..19 = inner1024
I=$SLURM_ARRAY_TASK_ID
if [ $I -lt 10 ]; then
  TAG=inner128
  D_INNER=128
  K_IDX=$I
  DIR_PATTERN="K${KS[$K_IDX]}_res2_inner128"
else
  TAG=inner1024
  D_INNER=1024
  K_IDX=$((I - 10))
  DIR_PATTERN="K${KS[$K_IDX]}_res2_inner1024"
fi
K=${KS[$K_IDX]}

CKPT_DIR=${ROOT}/results/v22_res2_closedloop/${DIR_PATTERN}
CKPT=${CKPT_DIR}/v13_residual_SWA_step3k-8k.pkl
if [ ! -f "${CKPT}" ]; then
  echo "SKIP: missing ${CKPT} (probably still training)"; exit 0
fi

OUT_DIR=${ROOT}/results/v22_res2_closedloop/Kscan_SWA_eval
mkdir -p ${OUT_DIR}
OUT_JSON=${OUT_DIR}/K${K}_${TAG}_SWA_step3k-8k_cold_full_zero.json

echo "=== eval K=${K} ${TAG} d_inner=${D_INNER} SWA ==="
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
