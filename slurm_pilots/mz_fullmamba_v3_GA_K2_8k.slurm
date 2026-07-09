#!/usr/bin/env bash
# v3 fast-data K=2 phase, resume from v2-GA K=1 step 4000 ckpt -> step 8000.
#
# v3 difference vs v2: train_ds and eval_ds are eager-loaded into RAM at
# startup (one .load() call after prepare_dataset_for_task), so per-step
# build_batch_from_indices no longer triggers zarr lazy reads. Expected
# step time: ~4-6s vs v2's ~13s.
#
# Memory: 160 GB (vs v2's 96 GB) to accommodate the eager-loaded ~63 GB
# data + JAX buffers + safety margin.

#SBATCH --job-name=mz-fm-v3-K2
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --constraint=gpu80
#SBATCH --time=09:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v3_K2_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/mz_fm_v3_K2_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v3/train_mz_fullmamba_v3.py
cd /home/lm8598/Weather_Global_experiments

RESULTS=/home/lm8598/Weather_Global_experiments/results/mz_residual_memory
SRC_DIR="${RESULTS}/mz_fullmamba_v3_PARITY_K1_3500to6k"
DST_DIR="${RESULTS}/mz_fullmamba_v3_GA_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_8k"
mkdir -p "${DST_DIR}"

if [[ ! -f "${SRC_DIR}/mz_residual_step4000.pkl" ]]; then
  echo "ERROR: K=1 step 4000 ckpt not found at ${SRC_DIR}/mz_residual_step4000.pkl"
  exit 1
fi

cp "${SRC_DIR}/train_log.json" "${DST_DIR}/train_log.json"
cp "${SRC_DIR}/eval_log.json" "${DST_DIR}/eval_log.json"

${PYTHON} ${SCRIPT} \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --baseline-ckpt "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${RESULTS}" \
  --run-name mz_fullmamba_v3_GA_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_8k \
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
