#!/usr/bin/env bash
# v3 K=8 phase RESUME from step 9000 (which is the K=8 ckpt that got saved
# before the previous job 7558363 was accidentally cancelled). Continues to
# step 11000 (2000 more K=8 steps).
#
# Submitted under --account=biao per user instruction.

#SBATCH --job-name=mz-fm-v3-K8-grp
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=12:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v3_K8_grp_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v3_K8_grp_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v3/train_mz_fullmamba_v3.py
cd /home/lm8598/Weather_Global_experiments

RESULTS=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory
DST_DIR="${RESULTS}/mz_fullmamba_v3_GA_K8_seg4_grouploss_LW1_step8500to11000"
RESUME_CKPT="${DST_DIR}/mz_residual_step9000.pkl"

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "ERROR: K=8 step 9000 ckpt not found at ${RESUME_CKPT}"
  exit 1
fi

# Train_log.json + eval_log.json already exist in DST_DIR with K-chain history
# through step 9000; trainer resume logic appends new entries. Do NOT overwrite.

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${RESULTS}" \
  --run-name mz_fullmamba_v3_GA_K8_seg4_grouploss_LW1_step8500to11000 \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h \
  --segment-steps 4 \
  --target-steps 8 --train-mode target_rollout \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --full-mamba --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --specialist-heads \
  --anchor-as-batch \
  --anchor-as-batch-min-k 2 \
  --use-group-loss \
  --upper-loss-weight 1.0 \
  --mslp-loss-weight 1.0 \
  --small-surface-loss-weight 0.1 \
  --loss-lead-weight-power 1.0 \
  --max-steps 11000 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${RESUME_CKPT}" \
  --resume-step 9000
