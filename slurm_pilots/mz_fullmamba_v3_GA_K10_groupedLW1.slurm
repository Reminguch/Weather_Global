#!/usr/bin/env bash
# v3 K=10 phase from K=8 step 11000 ckpt -> step 13500 (2500 K=10 training steps).
#
# K-curriculum continuation:
#   K=2 -> K=4: hidden contrib 0% -> 31%
#   K=4 -> K=6: K=6 plateau Z latw +2.10% vs K=4's +1.65%
#   K=6 -> K=8: extends SSM time axis to 8 (7 self-feed steps within anchor)
#   K=8 -> K=10: pushes to 10 (9 self-feed steps); does +60h Z keep climbing
#                or saturate? Approaches paper's K=12 endpoint.
#
# Same loss / arch as K=8 (verified non-regressing through curriculum):
#   --use-group-loss --upper-loss-weight 1.0 --mslp-loss-weight 1.0
#   --small-surface-loss-weight 0.1 --loss-lead-weight-power 1.0
#
# Memory: K=10 with segment_steps=4 -> [T=10, B=S=4, lat, lon, F].
# T*B = 40, ~25% more than K=8 (seg=4 -> 32). Safe on gpu80.
# (v3 trainer hard-constrains segment_steps >= 4, so seg=3 is rejected.)
# Step rate at this T·B: ~13-15s/step -> 2500 steps ≈ 10h.

#SBATCH --job-name=mz-fm-v3-K10-grp
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=12:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v3_K10_grp_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v3_K10_grp_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v3/train_mz_fullmamba_v3.py
cd /home/lm8598/Weather_Global_experiments

RESULTS=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory
SRC_DIR="${RESULTS}/mz_fullmamba_v3_GA_K8_seg4_grouploss_LW1_step8500to11000"
DST_DIR="${RESULTS}/mz_fullmamba_v3_GA_K10_seg3_grouploss_LW1_step11000to13500"
mkdir -p "${DST_DIR}"

if [[ ! -f "${SRC_DIR}/mz_residual_step11000.pkl" ]]; then
  echo "ERROR: K=8 step 11000 ckpt not found at ${SRC_DIR}"
  exit 1
fi

# Pre-seed logs so K=1->K=2->K=4->K=6->K=8 (steps 0..11000) followed by
# K=10 (11000..13500) appear as one continuous curriculum curve.
cp "${SRC_DIR}/train_log.json" "${DST_DIR}/train_log.json"
cp "${SRC_DIR}/eval_log.json"  "${DST_DIR}/eval_log.json"

${PYTHON} -u ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${RESULTS}" \
  --run-name mz_fullmamba_v3_GA_K10_seg3_grouploss_LW1_step11000to13500 \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h \
  --segment-steps 4 \
  --target-steps 10 --train-mode target_rollout \
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
  --max-steps 13500 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${SRC_DIR}/mz_residual_step11000.pkl" \
  --resume-step 11000
