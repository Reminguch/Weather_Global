#!/usr/bin/env bash
# v2 K=1 from random init, 4000 steps. Three v2 mods all ON:
#   --specialist-heads        (Mod G)
#   --anchor-as-batch         (Mod A)
#   --loss-lead-weight-power 0.0   (Mod C, p=0 trivially uniform at K=1)
#
# This is the head ckpt for the v2 K-curriculum chain. Compare against
# v1 K=1 best (mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_4k_cont
# step 4000, baseline_overall_MAE Δ% = +1.22%, Z RMSE Δ% = +1.67%, MSLP +1.31%).
#
# Expected: K=1 v2 ≈ K=1 v1 best on Z (memory unlock not active at K=1),
# possibly slightly better on surface variables (specialist heads frees them
# from upper-air gradient pressure, though K=1 doesn't have severe surface
# degradation to begin with). Real differentiation expected at K>=4.

#SBATCH --job-name=mz-fm-v2-K1-4k
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=09:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v2_K1_4k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v2_K1_4k_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
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
  --run-name mz_fullmamba_v2_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K1_4k \
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
  --loss-lead-weight-power 0.0 \
  --max-steps 4000 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16
