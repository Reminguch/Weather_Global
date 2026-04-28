#!/usr/bin/env bash
# v2-GA K=1 from scratch, 4000 steps.
#
# Flags:
#   --specialist-heads            (Mod G)
#   --anchor-as-batch             (Mod A requested)
#   --anchor-as-batch-min-k 2     (Patch 1: K=1 -> effective=False, legacy
#                                   layout, preserves v1 stateful cycling)
#   --loss-lead-weight-power 0.0  (Mod C: K=1 has only 1 lead, weight is
#                                   trivially 1; harmless)
#
# Expected [layout] log line at startup:
#   requested_anchor_as_batch=True  min_k=2  target_steps=1
#   effective_anchor_as_batch=False  allow_tf_at_t0=False
#
# This is the head ckpt for the v2-GA K-curriculum chain. Body trains under
# legacy K=1 layout (compatible with v1 K=1); only Mod G distinguishes.

#SBATCH --job-name=mz-fm-v2GA-K1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=09:00:00
#SBATCH --output=/home/lm8598/Fermihubbardnumeric/logs/mz_fm_v2GA_K1_%j.out
#SBATCH --error=/home/lm8598/Fermihubbardnumeric/logs/mz_fm_v2GA_K1_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Fermihubbardnumeric/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba/train_mz_fullmamba_v2.py
cd /home/lm8598/Weather_Global_experiments

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \
  --run-name mz_fullmamba_v2_GA_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K1_4k \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h --segment-steps 16 \
  --target-steps 1 --train-mode teacher \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --full-mamba --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --specialist-heads \
  --anchor-as-batch \
  --anchor-as-batch-min-k 2 \
  --loss-lead-weight-power 0.0 \
  --max-steps 4000 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16
