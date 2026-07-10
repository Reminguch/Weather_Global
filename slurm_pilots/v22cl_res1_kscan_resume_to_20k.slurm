#!/usr/bin/env bash
# Wave 2: Resume res=1 closed-loop K-scan from step 14000 → step 20000 (6000 more steps).
# Walltime: 6000 × ~6.6s = 11h.
#SBATCH --job-name=v22cl-r1-resume20k
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=22:00:00
#SBATCH --array=0-10%6
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22cl_r1_resume20k_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22cl_r1_resume20k_t%a_%A.err
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
OUT=${ROOT}/results/v22_res1_closedloop_fresh

K_LABELS=(2 4 6 8 10 12 14 16 18 20 22)
K=${K_LABELS[$SLURM_ARRAY_TASK_ID]}
AR_TAIL=${K}
RESUME_CKPT=${OUT}/K${K}_closedloop_sg_fresh_dmckpt/v13_residual_step14000.pkl

[ -f "${RESUME_CKPT}" ] || { echo "ERR: missing resume ckpt ${RESUME_CKPT}"; exit 1; }

echo "=== Resume v22cl res=1 K=${K} from step 14000 → 20000 ==="

${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${DM_CKPT}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K${K}_closedloop_sg_fresh_dmckpt \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 20000 --checkpoint-every 1000 \
  --start-step 14001 \
  --resume-from "${RESUME_CKPT}" \
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 0 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K ${AR_TAIL} \
  --feedback-mode closed_loop_sg \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2

echo "v22cl res=1 K=${K} step 20000 done"
