#!/usr/bin/env bash
# v22 res=2 K=18 closed-loop-SG architecture screen (6 configs).
# inner128 is already trained at K18_res2_closedloop_sg_fresh (2000 steps done).
#
# 7-config screen (inner128 = existing baseline, 6 new here):
#   idx 0: inner256   d_inner=256 d_state=16 layers=2 d_conv=4  (width 2x)
#   idx 1: inner512   d_inner=512 d_state=16 layers=2 d_conv=4  (width 4x)
#   idx 2: inner1024  d_inner=1024 d_state=16 layers=2 d_conv=4 (width 8x)
#   idx 3: state32    d_inner=128 d_state=32 layers=2 d_conv=4  (SSM memory 2x)
#   idx 4: depth4     d_inner=128 d_state=16 layers=4 d_conv=4  (depth 2x)
#   idx 5: conv8      d_inner=128 d_state=16 layers=2 d_conv=8  (local temporal 2x)
#
# Same base ckpt as inner128:
#   GCv1 K=3 cascade res=2 mesh=4 width=512 mp=6 @ step 52000.
#SBATCH --job-name=v22cl-res2-arch-screen
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_res2_arch_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_res2_arch_t%a_%A.err
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
[ -f "${GCV1_K3}" ] || { echo "ERR: missing GCv1 K3 ckpt"; exit 1; }

# Config table indexed by SLURM_ARRAY_TASK_ID
NAMES=(inner256 inner512 inner1024 state32 depth4 conv8)
D_INNERS=(256 512 1024 128 128 128)
D_STATES=(16 16 16 32 16 16)
LAYERS=(2 2 2 2 4 2)
D_CONVS=(4 4 4 4 4 8)

I=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$I]}
D_INNER=${D_INNERS[$I]}
D_STATE=${D_STATES[$I]}
LAYER=${LAYERS[$I]}
D_CONV=${D_CONVS[$I]}

OUT=${ROOT}/results/v22_res2_closedloop/K18_res2_${NAME}
mkdir -p ${OUT}

# nvidia-smi peak-mem tracker
(while true; do
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv >> ${OUT}/nvsmi.log 2>&1 || true
  sleep 60
done) &
NVSMI_PID=$!

echo "=== v22cl res=2 K=18 arch-screen  cfg=${NAME}  d_inner=${D_INNER}  d_state=${D_STATE}  layers=${LAYER}  d_conv=${D_CONV} ==="
${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${GCV1_K3}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K18_res2_${NAME} \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 2000 --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 18 \
  --feedback-mode closed_loop_sg \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size ${D_INNER} \
  --temporal-d-inner ${D_INNER} \
  --temporal-d-state ${D_STATE} \
  --temporal-d-conv ${D_CONV} \
  --temporal-dt-rank auto \
  --temporal-layers ${LAYER}

kill $NVSMI_PID 2>/dev/null || true
echo "=== ${NAME} done ==="
