#!/usr/bin/env bash
# v22 res=2 K=18 arch-screen: EXTEND all 7 configs from step 2000 → 10000.
# Same base ckpt (GCv1 K=3 res2 step 52000). Resumes from each config's own
# step 2000 residual pkl. Saves ckpt every 500 steps → 16 new ckpts per config.
#SBATCH --job-name=v22cl-res2-arch-extend
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=10:00:00
#SBATCH --array=0-6
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_res2_arch_ext_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_res2_arch_ext_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
PREPARED_ROOT=/scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/dataset/prepared_stream/res2
RESIDUAL_ROOT=${ROOT}/data/graphcast/graphcast/dataset/precomputed_residuals/v22_res2_GCv1K3_synthetic

# 7 configs (order matters):
NAMES=(inner128 inner256 inner512 inner1024 state32 depth4 conv8)
D_INNERS=(128 256 512 1024 128 128 128)
D_STATES=(16 16 16 16 32 16 16)
LAYERS=(2 2 2 2 2 4 2)
D_CONVS=(4 4 4 4 4 4 8)

# inner128 previous run dir is a different name — read step-2000 from there
BASE128_PREV_DIR=${ROOT}/results/v22_res2_closedloop/K18_res2_closedloop_sg_fresh

I=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$I]}
D_INNER=${D_INNERS[$I]}
D_STATE=${D_STATES[$I]}
LAYER=${LAYERS[$I]}
D_CONV=${D_CONVS[$I]}

OUT_PARENT=${ROOT}/results/v22_res2_closedloop
OUT=${OUT_PARENT}/K18_res2_${NAME}
if [ "${NAME}" = "inner128" ]; then
  RESUME_CKPT=${BASE128_PREV_DIR}/v13_residual_step2000.pkl
else
  RESUME_CKPT=${OUT}/v13_residual_step2000.pkl
fi
[ -f "${RESUME_CKPT}" ] || { echo "ERR: missing ${RESUME_CKPT}"; exit 1; }

# For inner128, seed the flat dir with prior ckpts so extension keeps them.
mkdir -p ${OUT}
if [ "${NAME}" = "inner128" ] && [ ! -f "${OUT}/v13_residual_step2000.pkl" ]; then
  cp ${BASE128_PREV_DIR}/v13_residual_step*.pkl ${OUT}/ 2>/dev/null || true
fi

echo "=== EXTEND ${NAME}  d_inner=${D_INNER} d_state=${D_STATE} layers=${LAYER} d_conv=${D_CONV} 2000->10000 ==="
echo "resume from ${RESUME_CKPT}"
# Note: train_mz_v20.py does out_dir/run_name/. To keep flat layout, out-dir=parent, run-name=K18_res2_<name>.
${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${GCV1_K3}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT_PARENT}" \
  --run-name K18_res2_${NAME} \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 10000 --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 18 \
  --feedback-mode closed_loop_sg \
  --resume-from "${RESUME_CKPT}" \
  --start-step 2001 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size ${D_INNER} \
  --temporal-d-inner ${D_INNER} \
  --temporal-d-state ${D_STATE} \
  --temporal-d-conv ${D_CONV} \
  --temporal-dt-rank auto \
  --temporal-layers ${LAYER}

echo "=== ${NAME} extension done ==="
