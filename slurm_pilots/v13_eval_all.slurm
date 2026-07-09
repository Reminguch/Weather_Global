#!/usr/bin/env bash
# Eval all v13 20K ckpts at step 1k...20k.
# v13 uses v9's GCResidualWithZeroHead architecture (full DeepMind GC2 +
# Mamba interleaved + zero-init head), so we use scripts/training/full_mamba_v9/eval_v9.py.

#SBATCH --job-name=v13-eval-all
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v13_eval_all_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v13_eval_all_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v9/eval_v9.py
cd /home/lm8598/Weather_Global_experiments

CKPT_BASE=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v13_v9arch_fast/v13_20k
EVAL_OUT_DIR=${CKPT_BASE}/eval_results
mkdir -p "${EVAL_OUT_DIR}"

for STEP in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000; do
  CKPT="${CKPT_BASE}/v13_residual_step${STEP}.pkl"
  OUT="${EVAL_OUT_DIR}/eval_step${STEP}.json"
  if [[ -f "${OUT}" ]]; then
    echo "[skip] step${STEP} already evaluated"
    continue
  fi
  if [[ ! -f "${CKPT}" ]]; then
    echo "[skip] missing ckpt step${STEP}"
    continue
  fi
  echo "============= EVAL v13 step ${STEP} ============="
  ${PYTHON} -u ${SCRIPT} \
    --ckpt "${CKPT}" \
    --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
    --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
    --residual-msg-steps 2 \
    --temporal-location mesh_processor_interleaved \
    --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 4 \
    --temporal-dt-rank auto --temporal-layers 2 \
    --n-samples 32 \
    --out-json "${OUT}"
done
echo "all done"
