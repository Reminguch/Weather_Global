#!/usr/bin/env bash
# Eval all 7 arch-screen configs at res=2 K=18 CL. Reads each config's step 2000
# ckpt and reports paper-weighted MSE + per-var improvement vs GCv1 K=3 baseline.
#
# Uses eval_v22_clean.py --eval-mode cold_full --residual-state-init zero for
# apples-to-apples with previous v22cl_r1 eval convention.
#SBATCH --job-name=v22cl-res2-arch-eval
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --array=0-6
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_res2_arch_eval_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_res2_arch_eval_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
# Data path — res=2 wb2 or existing 2015-2022 zarr
DATA=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr
# Note: eval_v22_clean uses resolution=2.0 will coarsen internally? Check needed.
# Actually the res=2 training pipeline uses the same 1° zarr and processes at res=2 spatially.
# But for eval we need a res=2 compatible input. Look at how v22_res2_kscan_eval_cold_bp does it.

NAMES=(inner128 inner256 inner512 inner1024 state32 depth4 conv8)
D_INNERS=(128 256 512 1024 128 128 128)
D_STATES=(16 16 16 16 32 16 16)
LAYERS=(2 2 2 2 2 4 2)
D_CONVS=(4 4 4 4 4 4 8)

I=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$I]}
D_INNER=${D_INNERS[$I]}
D_STATE=${D_STATES[$I]}
LAYER=${LAYERS[$I]}
D_CONV=${D_CONVS[$I]}

# inner128 lives in K18_res2_closedloop_sg_fresh; others in K18_res2_<name>
if [ "${NAME}" = "inner128" ]; then
  CKPT_DIR=${ROOT}/results/v22_res2_closedloop/K18_res2_closedloop_sg_fresh
else
  CKPT_DIR=${ROOT}/results/v22_res2_closedloop/K18_res2_${NAME}
fi
CKPT=${CKPT_DIR}/v13_residual_step2000.pkl
[ -f "${CKPT}" ] || { echo "ERR: missing ${CKPT}"; exit 1; }

OUT_DIR=${ROOT}/results/v22_res2_closedloop/arch_eval_step2000
mkdir -p ${OUT_DIR}
OUT_JSON=${OUT_DIR}/${NAME}_step2000_cold_full_zero.json

echo "=== eval ${NAME}  d_inner=${D_INNER}  step=2000 ==="
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
  --temporal-d-state ${D_STATE} --temporal-d-conv ${D_CONV} \
  --temporal-dt-rank auto --temporal-layers ${LAYER} \
  --eval-mode cold_full --residual-state-init zero \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
  --out-json "${OUT_JSON}"
echo "=== done ${NAME} ==="
