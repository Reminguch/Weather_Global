#!/usr/bin/env bash
# v3 K=4 phase from v3 K=2 step 5500 ckpt -> step 7500 (2000 K=4 training steps).
#
# Memory-safe (no eager-load) — Mod A's anchor-as-batch [T=4, B=8] is the
# memory hot path; eager-load on top of that is what OOM'd K=2 at step 5800.
# Lazy zarr loads cost ~13s/step but no OOM risk.
#
# K=4 is the first phase where Mod A unlock memory should plausibly show up
# (T=4 SSM time axis = 3 self-feed steps after anchor, exponential kernel
# has nontrivial accumulation space). Compare within-phase Z latw climb
# against v1 K=4 fresh trajectory (step 5200->6000 in v1 K2_proc2_9k or
# K4_7k chains).
#
# eval-time AR is implicit (eval_autoregressive=True default), gives latw
# both for TF and AR.

#SBATCH --job-name=mz-fm-v3-K4
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=09:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v3_K4_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v3_K4_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v3/train_mz_fullmamba_v3.py
cd /home/lm8598/Weather_Global_experiments

RESULTS=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory
SRC_DIR="${RESULTS}/mz_fullmamba_v3_GA_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_8k"
DST_DIR="${RESULTS}/mz_fullmamba_v3_GA_paperckpt_r1_in2_seg8_meshed_m5_h128_ds16_fullvars_K4_7500"
mkdir -p "${DST_DIR}"

if [[ ! -f "${SRC_DIR}/mz_residual_step5500.pkl" ]]; then
  echo "ERROR: K=2 step 5500 ckpt not found at ${SRC_DIR}/mz_residual_step5500.pkl"
  exit 1
fi

# Pre-seed train+eval logs so the full chain step 0..5500 trajectory is
# preserved and 5501..7500 entries append after.
cp "${SRC_DIR}/train_log.json" "${DST_DIR}/train_log.json"
cp "${SRC_DIR}/eval_log.json" "${DST_DIR}/eval_log.json"

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${RESULTS}" \
  --run-name mz_fullmamba_v3_GA_paperckpt_r1_in2_seg8_meshed_m5_h128_ds16_fullvars_K4_7500 \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h \
  --segment-steps 8 \
  --target-steps 4 --train-mode target_rollout \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --full-mamba --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --specialist-heads \
  --anchor-as-batch \
  --anchor-as-batch-min-k 2 \
  --loss-lead-weight-power 0.0 \
  --max-steps 7500 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${SRC_DIR}/mz_residual_step5500.pkl" \
  --resume-step 5500
