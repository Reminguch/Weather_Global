#!/usr/bin/env bash
# K=2 with 2-layer spatial GNN processor between mesh encoder and Mamba SSM.
# Resume from FullMamba K=2 step 5000 (no-processor ckpt); processor sub-tree
# gets fresh init via --allow-partial-resume. Trains 1500 more K=2 steps to
# 6500 — the goal is to see whether spatial message passing on the icosphere
# unlocks the synoptic-scale variables (MSLP, Z) that pure-temporal Mamba
# couldn't reach. Independent of the K=4 line (job 7359536), runs in parallel.

#SBATCH --job-name=mz-K2-proc2
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=10:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_K2_proc2_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_K2_proc2_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba/train_mz_fullmamba.py
cd /home/lm8598/Weather_Global_experiments

SRC_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_5k
DST_NAME=mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_proc2_6500
DST_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/${DST_NAME}
mkdir -p "${DST_DIR}"

# Pre-seed logs so the new run continues 0..5000 (no-processor) followed by
# 5000..6500 (with processor) as one continuous curve.
cp "${SRC_DIR}/train_log.json" "${DST_DIR}/train_log.json"
cp "${SRC_DIR}/eval_log.json"  "${DST_DIR}/eval_log.json"

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir /home/lm8598/Weather_Global_experiments/results/mz_residual_memory \
  --run-name ${DST_NAME} \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 1.0 --mesh-size 5 --input-duration 12h --segment-steps 16 \
  --target-steps 2 --train-mode target_rollout \
  --hidden-size 128 --layers 1 \
  --meshed --mz-mesh-size 5 --n-grid-neighbors 6 --n-mesh-neighbors 3 \
  --full-mamba --d-state 16 --expand 2 \
  --a-log-init-min -3.0 --a-log-init-max -0.1 \
  --full-variables \
  --processor-layers 2 \
  --max-steps 6500 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 100 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${SRC_DIR}/mz_residual_step5000.pkl" \
  --resume-step 5000 \
  --allow-partial-resume
