#!/usr/bin/env bash
# Closed-loop sg K-scan cascade at res=1, starting from ORIGINAL v22 K=1 ckpt.
# Matches v22 paper structure exactly: K=2..K=22 all cascade from
#   /scratch/.../v22/K1_from_v15v2_20k/v13_residual_step23000.pkl (open-loop trained).
# Only difference from v22 paper: K≥2 phase uses --feedback-mode closed_loop_sg.
#SBATCH --job-name=v22cl-r1-kscan-v22K1
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=22:00:00
#SBATCH --array=0-10%6
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_r1_kscan_v22K1_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_r1_kscan_v22K1_t%a_%A.err
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
# Original v22 K=1 ckpt (open-loop trained) — same starting point as v22 paper cascade
V22_K1_CKPT=${ROOT}/results/v22/K1_from_v15v2_20k/v13_residual_step23000.pkl
OUT=${ROOT}/results/v22_res1_closedloop

K_LABELS=(2 4 6 8 10 12 14 16 18 20 22)
K=${K_LABELS[$SLURM_ARRAY_TASK_ID]}
AR_TAIL=${K}   # v22 convention for K>=2: ar_tail_K = K

[ -f "${V22_K1_CKPT}" ] || { echo "ERR: missing v22 K=1 ckpt"; exit 1; }
mkdir -p ${OUT}

echo "==============================="
echo "=== v22cl res=1 K=${K} (ar_tail=${AR_TAIL})"
echo "=== from ORIGINAL v22 K=1 (open-loop) step 23k"
echo "==============================="

${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${DM_CKPT}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K${K}_closedloop_sg_from_v22K1_23k \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 26000 --checkpoint-every 500 \
  --start-step 23001 \
  --resume-from "${V22_K1_CKPT}" \
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K ${AR_TAIL} \
  --feedback-mode closed_loop_sg \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2

echo "v22cl res=1 K=${K} (from v22K1) done"
