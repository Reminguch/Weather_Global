#!/usr/bin/env bash
# v22closed H=256 K=22 SMOKE (200 steps): verify memory + step_time + no NaN
# before committing to 30h full run.
#
# Changes vs v22closed_sg_K22_production.slurm:
#   --temporal-hidden-size 128 → 256   (main scaling axis)
#   --max-steps 20000 → 200            (smoke)
#   --checkpoint-every 1000 → 500      (any ckpt is bonus)
#   --time=30:00:00 → 01:30:00
#
# Everything else identical (SG-feedback, K=22, bf16, batch=1, lr=1e-4).
#SBATCH --job-name=v22closed-H256-smoke
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22closed_H256_smoke_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22closed_H256_smoke_%j.err
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
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K22_H256_smoke

# nvidia-smi snapshot at start (background loop every 30s) → helps us see peak mem
(while true; do
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv >> ${OUT}/nvsmi.log 2>&1 || true
  sleep 30
done) &
NVSMI_PID=$!
mkdir -p ${OUT}

${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name v22closed_H256_K22_smoke \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 200 --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 500 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 22 \
  --feedback-stop-gradient \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 256 \
  --temporal-d-inner 256 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2

kill $NVSMI_PID 2>/dev/null || true
echo "=== smoke done ==="
