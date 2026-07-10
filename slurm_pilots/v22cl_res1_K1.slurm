#!/usr/bin/env bash
# Closed-loop sg K=1 training at res=1, mirroring v22 K1_from_v15v2_20k structure.
# Resume from v15v2 step 20000, train 3000 steps (steps 20001..23000) with
# --feedback-mode closed_loop_sg, --ar-tail-K 0 (matches v22 K=1 paper convention).
# Output: ckpt at step 23000 to feed the K=2..K=22 cascade.
#SBATCH --job-name=v22cl-r1-K1
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=22:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_r1_K1_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_r1_K1_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=${ROOT}/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RESIDUAL_ROOT=${ROOT}/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2
V15V2_CKPT=${ROOT}/results/v15_v13arch_7yr_v2/v15_20k_v2/v13_residual_step20000.pkl
OUT=${ROOT}/results/v22_res1_closedloop

mkdir -p ${OUT}
[ -f "${V15V2_CKPT}" ] || { echo "ERR: missing v15v2 base ckpt ${V15V2_CKPT}"; exit 1; }
[ -f "${DM_CKPT}" ]    || { echo "ERR: missing DeepMind GC ckpt"; exit 1; }

${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${DM_CKPT}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K1_closedloop_sg_from_v15v2_20k \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 23000 --checkpoint-every 500 \
  --start-step 20001 \
  --resume-from "${V15V2_CKPT}" \
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 0 \
  --feedback-mode closed_loop_sg \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2

echo "v22cl K=1 res=1 closed-loop sg done"
