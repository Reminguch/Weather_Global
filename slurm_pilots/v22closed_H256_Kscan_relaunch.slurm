#!/usr/bin/env bash
# Relaunch res=1 K=4/8/12/16 H=256 from step 19000 -> 20000 (finish last 1000 steps
# that TIMEOUT'd at 36h wallclock limit).
#SBATCH --job-name=v22closed-H256-Kscan-relaunch
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22closed_H256_relaunch_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22closed_H256_relaunch_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_CPP_MIN_LOG_LEVEL=1
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v22closed/train_mz_v22closed.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RESIDUAL_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2

KS=(4 8 12 16)
K=${KS[$SLURM_ARRAY_TASK_ID]}

OUT_PARENT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K${K}_H256_fresh_20k
# Note: previous nested layout — files in <parent>/v22closed_H256_K{K}_fresh_20k/
PREV_INNER=${OUT_PARENT}/v22closed_H256_K${K}_fresh_20k
RESUME_CKPT=${PREV_INNER}/v13_residual_step19000.pkl
[ -f "${RESUME_CKPT}" ] || { echo "ERR: missing ${RESUME_CKPT}"; exit 1; }

echo "=== Relaunch v22closed K=${K} H=256 resume from step 19000 -> 20000 ==="
${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT_PARENT}" \
  --run-name v22closed_H256_K${K}_fresh_20k \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 20000 --checkpoint-every 1000 \
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 500 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K ${K} \
  --feedback-stop-gradient \
  --resume-from "${RESUME_CKPT}" \
  --start-step 19001 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 256 \
  --temporal-d-inner 256 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2

echo "=== K=${K} relaunch done ==="
