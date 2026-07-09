#!/usr/bin/env bash
# K=2 continued training to step 8000 (from step 5000): 3000 more K=2 steps
# to reach 4000 K=2 steps total, matching K=1's 4000-step training budget for
# fair K-curriculum comparison.
#
# Existing K=2 (job 7334399) was only trained for 1000 K=2 steps. Z RMSE TF
# trajectory at step 4200..5000 (+1.06% -> +1.13%, lat-w +1.49%) was still
# climbing; loss MA(50) was still decreasing -0.46% over the 1000 steps. All
# downstream curriculum (K=4, K=6, K=8) was launched from this undertrained
# K=2 ckpt -- so before redoing the K-curriculum chain, K=2 itself must be
# fully converged. After this finishes, we will branch K=4 / K=6 / K=8 fresh
# from K=2 step 8000.
#
# Strict resume (architecture unchanged), pure FullMamba (no processor).

#SBATCH --job-name=mz-fm-K2-8k
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=20:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_K2_8k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_K2_8k_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba/train_mz_fullmamba.py
cd /home/lm8598/Weather_Global_experiments

SRC_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_5k
DST_NAME=mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_8k
DST_DIR=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/${DST_NAME}
mkdir -p "${DST_DIR}"

# Pre-seed logs so 0..5000 (K=1+initial K=2) plus 5000..8000 (continued K=2)
# appear as one continuous curve.
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
  --max-steps 8000 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 100 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${SRC_DIR}/mz_residual_step5000.pkl" \
  --resume-step 5000
