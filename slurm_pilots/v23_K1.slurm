#!/usr/bin/env bash
# v23 K=1 spinup — same as v22 K=1 except --temporal-d-conv 8 (was 4).
# Resume from v15 step20000 baseline; train 3000 steps with K=1 (ar-tail-K=0).
#SBATCH --job-name=v23-K1
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=08:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v23_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v23_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v20/train_mz_v20.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RESIDUAL_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v23
# v15 step20000 had d_conv=4 Mamba weights + state — can't load directly into a d_conv=8 model.
# We use strip_state_v15.py to pad conv1d kernels [128,4]->[128,8] (front-zero) and drop state.
# See scripts/training/full_mamba_v23/strip_state_v15.py for details.
RESUME_CKPT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v23_helpers/v15_step20000_for_dconv8.pkl

# Guard: require pre-processed warm-start ckpt to exist.
if [ ! -f "${RESUME_CKPT}" ]; then
  echo "ERROR: missing warm-start ckpt: ${RESUME_CKPT}" 1>&2
  echo "Generate it with: python scripts/training/full_mamba_v23/strip_state_v15.py" 1>&2
  exit 1
fi

${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name K1_from_v15v2_20k \
  --resume-from "${RESUME_CKPT}" \
  --start-step 20001 \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 23000 --checkpoint-every 500 \
  --lr 1e-4 --weight-decay 1e-4 --seed 23 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 200 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 0 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-state 16 --temporal-d-conv 8 --temporal-dt-rank auto \
  --temporal-layers 2
