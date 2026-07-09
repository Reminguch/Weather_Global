#!/usr/bin/env bash
#SBATCH --job-name=v22closed-K18-autoresume
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=24:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22closed_K18_autoresume_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22closed_K18_autoresume_%j.err
#SBATCH --dependency=afterany:9203729
set -euo pipefail

INNER_DIR="/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K18_sg_fresh_20k/v22closed_sg_K18_fresh_20k"
# K=22 production has different inner name
if [ "18" = "22" ]; then
  INNER_DIR="/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed/K18_sg_fresh_20k/v22closed_sg_K22_fresh_20k"
  # Actually K=22 prod uses run-name "v22closed_sg_K22_fresh_20k"
  # so out_dir/run_name = K22_fresh_20k/v22closed_sg_K22_fresh_20k
fi

# Idempotent check: skip if step20000.pkl exists
if [ -f "${INNER_DIR}/v13_residual_step20000.pkl" ]; then
  echo "K=18: step20000.pkl exists at ${INNER_DIR}, parent succeeded. SKIP."
  exit 0
fi

# Find latest saved ckpt
LATEST=$(ls "${INNER_DIR}"/v13_residual_step*.pkl 2>/dev/null | sed 's/.*step//;s/\.pkl//' | sort -n | tail -1)
if [ -z "${LATEST}" ]; then
  echo "K=18: NO ckpt in ${INNER_DIR}. Cannot resume." 1>&2
  exit 1
fi
RESUME_CKPT="${INNER_DIR}/v13_residual_step${LATEST}.pkl"
START_STEP=$((LATEST + 1))
echo "K=18: resuming from step ${LATEST} (${RESUME_CKPT}) → 20000"

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

${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed \
  --run-name K18_sg_fresh_20k \
  --resume-from "${RESUME_CKPT}" \
  --start-step ${START_STEP} \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 20000 --checkpoint-every 1000 \
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 0 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 18 \
  --feedback-stop-gradient \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2
