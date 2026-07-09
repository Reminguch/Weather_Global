#!/usr/bin/env bash
# v2-GA K=1 RESUME from step 2000 ckpt -> step 4000.
#
# Original 7434804 TIMEOUT'd at step 2000 (8.5h elapsed, ~15 sec/step).
# Continue from step 2000 to finish remaining 2000 steps. Same flags.
#
# Wall time 7h is generous: 2000 steps × 15s = ~8.3h. Setting --time=08:30:00
# to leave margin. If it timeouts again we still get an extra 1500-2000 steps
# of progress beyond step 2000.

#SBATCH --job-name=mz-fm-v2GA-K1r
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=08:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v2GA_K1_resume_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v2GA_K1_resume_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba/train_mz_fullmamba_v2.py
cd /home/lm8598/Weather_Global_experiments

RUN_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_v2_GA_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K1_4k

if [[ ! -f "${RUN_DIR}/mz_residual_step2000.pkl" ]]; then
  echo "ERROR: step 2000 ckpt not found at ${RUN_DIR}/mz_residual_step2000.pkl"
  exit 1
fi

# Resume INTO the same run-dir so train_log / eval_log keep appending.
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
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${RUN_DIR}/mz_residual_step2000.pkl" \
  --resume-step 2000
