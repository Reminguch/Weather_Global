#!/usr/bin/env bash
# D5 — lead-weighted loss diagnostic. Resume from v3 K=4 step 7000 with
# --loss-lead-weight-power 1.0 (Mod C linear lead weighting). Trains 800
# more steps. Hypothesis: the K=4 plateau may come from uniform-lead loss
# being dominated by easy +6h short-lead errors, drowning out the harder
# +24h long-lead signal that's actually where Mamba memory should shine.
# If LW power=1.0 unsticks train loss OR pulls up +24h eval Δ%, uniform
# loss was the bottleneck.
#
# Same LR (1e-4) so the only changing variable is the lead-weight power.
# Goes to a separate run_dir to keep the original K=4 chain clean.

#SBATCH --job-name=mz-fm-v3-K4-LW1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v3_K4_LW1_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v3_K4_LW1_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v3/train_mz_fullmamba_v3.py
cd /home/lm8598/Weather_Global_experiments

RESULTS=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory
SRC_DIR="${RESULTS}/mz_fullmamba_v3_GA_paperckpt_r1_in2_seg8_meshed_m5_h128_ds16_fullvars_K4_7500"
DST_DIR="${RESULTS}/mz_fullmamba_v3_K4_LWpower1_step7000to7800_DIAG"
mkdir -p "${DST_DIR}"

cp "${SRC_DIR}/train_log.json" "${DST_DIR}/train_log.json"
cp "${SRC_DIR}/eval_log.json" "${DST_DIR}/eval_log.json"

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${RESULTS}" \
  --run-name mz_fullmamba_v3_K4_LWpower1_step7000to7800_DIAG \
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
  --loss-lead-weight-power 1.0 \
  --lr 1e-4 \
  --max-steps 7800 --eval-every 100 --eval-max-segments 8 --checkpoint-every 200 \
  --grad-clip 1.0 --warmup-steps 50 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${SRC_DIR}/mz_residual_step7000.pkl" \
  --resume-step 7000
