#!/usr/bin/env bash
# v3 K=6 phase from v3 K=4 step 7000 ckpt -> step 9500 (2500 K=6 training steps).
#
# K=4 (step 5500->7000) plateaued at train_loss ~0.184, eval Δ% bouncing in
# 1.4-1.7%. Train loss flat for 1000 steps + grad_norm 0.013 << grad_clip
# means the K=4 setting is optimization-saturated; bumping the rollout
# horizon to K=6 changes the training distribution (3 more self-feed steps
# inside each anchor), which is the v3 paper's whole "K-curriculum unlocks
# memory" mechanism. If Mamba hidden state is real memory, K=6 should pull
# more signal out than K=4 did. If not, the 7532119 ablation will tell us.
#
# Memory: K=4 used [T=4, B=S=8, lat, lon, F] under anchor_as_batch with
# segment_steps=8, T*B = 32. K=6 with segment_steps=5 gives [T=6, B=5, ...]
# i.e. T*B = 30 — slightly under K=4 budget so OOM risk is similar/lower.
# Lazy zarr loads (no eager-load) are the proven memory-safe path.

#SBATCH --job-name=mz-fm-v3-K6
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=12:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v3_K6_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v3_K6_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v3/train_mz_fullmamba_v3.py
cd /home/lm8598/Weather_Global_experiments

RESULTS=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory
SRC_DIR="${RESULTS}/mz_fullmamba_v3_GA_paperckpt_r1_in2_seg8_meshed_m5_h128_ds16_fullvars_K4_7500"
DST_DIR="${RESULTS}/mz_fullmamba_v3_GA_paperckpt_r1_in2_seg5_meshed_m5_h128_ds16_fullvars_K6_9500"
mkdir -p "${DST_DIR}"

if [[ ! -f "${SRC_DIR}/mz_residual_step7000.pkl" ]]; then
  echo "ERROR: K=4 step 7000 ckpt not found at ${SRC_DIR}/mz_residual_step7000.pkl"
  exit 1
fi

# Pre-seed train+eval logs so the full chain step 0..7000 trajectory is
# preserved and 7001..9500 entries append after.
cp "${SRC_DIR}/train_log.json" "${DST_DIR}/train_log.json"
cp "${SRC_DIR}/eval_log.json" "${DST_DIR}/eval_log.json"

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${RESULTS}" \
  --run-name mz_fullmamba_v3_GA_paperckpt_r1_in2_seg5_meshed_m5_h128_ds16_fullvars_K6_9500 \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h \
  --segment-steps 5 \
  --target-steps 6 --train-mode target_rollout \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --full-mamba --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --specialist-heads \
  --anchor-as-batch \
  --anchor-as-batch-min-k 2 \
  --loss-lead-weight-power 0.0 \
  --max-steps 9500 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${SRC_DIR}/mz_residual_step7000.pkl" \
  --resume-step 7000
