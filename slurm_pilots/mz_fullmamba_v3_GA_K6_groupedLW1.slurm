#!/usr/bin/env bash
# v3 K=6 phase with three-group loss + linear lead-weighting.
#
# Resumes from K=4 step 7000 ckpt. Loss recipe is the new "memory-focused"
# combo motivated by D1 + D3 + plateau analysis:
#   * --use-group-loss splits the channel-mean MSE into three groups:
#       upper (Z/T/q/u/v/w), mslp, small_surface (2mT/10m_uv/precip).
#     Group weights:
#       upper=1.0, mslp=1.0, small_surface=0.1
#     This attenuates the noisy small-surface gradient on the shared Mamba
#     body so the body's update direction prioritises the variables that
#     D3 ablation confirmed Mamba hidden state actually carries memory for.
#   * --loss-lead-weight-power 1.0 (Mod C linear) emphasises long leads,
#     since D1 showed Z/MSLP correction grows monotonically with lead time
#     (memory-accumulating signature) while precip/humidity correction
#     decays with lead.
#
# Memory budget: K=6 with segment_steps=5 -> [T=6, B=S=5, lat, lon, F].
# T*B = 30, vs K=4's 32, so similar memory footprint. Lazy zarr loads.
#
# train_log.json now also logs loss_upper / loss_mslp / loss_small_surface
# per step, so plateau decomposition is directly observable in the log
# (was the missing instrument that made the K=4 plateau ambiguous).

#SBATCH --job-name=mz-fm-v3-K6-grp
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu80
#SBATCH --time=12:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v3_K6_grp_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v3_K6_grp_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v3/train_mz_fullmamba_v3.py
cd /home/lm8598/Weather_Global_experiments

RESULTS=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory
SRC_DIR="${RESULTS}/mz_fullmamba_v3_GA_paperckpt_r1_in2_seg8_meshed_m5_h128_ds16_fullvars_K4_7500"
DST_DIR="${RESULTS}/mz_fullmamba_v3_GA_K6_seg5_grouploss_LW1_step7000to9500"
mkdir -p "${DST_DIR}"

if [[ ! -f "${SRC_DIR}/mz_residual_step7000.pkl" ]]; then
  echo "ERROR: K=4 step 7000 ckpt not found"
  exit 1
fi

cp "${SRC_DIR}/train_log.json" "${DST_DIR}/train_log.json"
cp "${SRC_DIR}/eval_log.json" "${DST_DIR}/eval_log.json"

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${RESULTS}" \
  --run-name mz_fullmamba_v3_GA_K6_seg5_grouploss_LW1_step7000to9500 \
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
  --use-group-loss \
  --upper-loss-weight 1.0 \
  --mslp-loss-weight 1.0 \
  --small-surface-loss-weight 0.1 \
  --loss-lead-weight-power 1.0 \
  --max-steps 9500 --eval-every 200 --eval-max-segments 8 --checkpoint-every 500 \
  --grad-clip 1.0 --warmup-steps 200 \
  --baseline-precision fp32 --precision bf16 \
  --resume-from "${SRC_DIR}/mz_residual_step7000.pkl" \
  --resume-step 7000
