#!/usr/bin/env bash
# v2-GA K=2 phase, resume from K=1 step 4000, train to step 8000 (4k more).
#
# This is the FIRST run where Mod A is actually effective:
#   target_steps=2 >= anchor_as_batch_min_k(2) -> effective_anchor_as_batch=True
#
# Expected [layout] log line at startup:
#   requested_anchor_as_batch=True  min_k=2  target_steps=2
#   effective_anchor_as_batch=True  allow_tf_at_t0=True
#
# Compare to v1 K=2 fresh which TIMEOUT'd at step 7200 with overall Δ% +1.14%
# (plateaued, MSLP collapsed +1.31% -> +0.13%). v2-GA K=2 is the test of
# whether Mod A + Mod G + tf-at-t0 actually unlocks within-phase decay.
#
# Pre-seeds the K=2 dst run-dir with K=1's train/eval logs so the full
# 0..4000 trajectory is preserved and 4000..8000 entries are appended.

#SBATCH --job-name=mz-fm-v2GA-K2
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=09:00:00
#SBATCH --output=/home/lm8598/Fermihubbardnumeric/logs/mz_fm_v2GA_K2_%j.out
#SBATCH --error=/home/lm8598/Fermihubbardnumeric/logs/mz_fm_v2GA_K2_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Fermihubbardnumeric/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba/train_mz_fullmamba_v2.py
cd /home/lm8598/Weather_Global_experiments

RESULTS=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory
SRC_DIR="${RESULTS}/mz_fullmamba_v2_GA_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K1_4k"
DST_DIR="${RESULTS}/mz_fullmamba_v2_GA_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_8k"
mkdir -p "${DST_DIR}"

if [[ ! -f "${SRC_DIR}/mz_residual_step4000.pkl" ]]; then
  echo "ERROR: K=1 step 4000 ckpt not found at ${SRC_DIR}/mz_residual_step4000.pkl"
  echo "Did the K=1 SLURM job complete successfully?"
  exit 1
fi

# Pre-seed train/eval logs so the full chain (step 0..8000) is in one place.
cp "${SRC_DIR}/train_log.json" "${DST_DIR}/train_log.json"
cp "${SRC_DIR}/eval_log.json" "${DST_DIR}/eval_log.json"

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${RESULTS}" \
  --run-name mz_fullmamba_v2_GA_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_8k \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h --segment-steps 16 \
  --target-steps 2 --train-mode target_rollout \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --full-mamba --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --specialist-heads \
  --anchor-as-batch \
  --anchor-as-batch-min-k 2 \
  --loss-lead-weight-power 0.0 \
  --max-steps 8000 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${SRC_DIR}/mz_residual_step4000.pkl" \
  --resume-step 4000
