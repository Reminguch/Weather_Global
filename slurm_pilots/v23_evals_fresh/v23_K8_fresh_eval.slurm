#!/usr/bin/env bash
# v23 K=8 fresh — 240h MSE eval. afterok on training job 9000472.
#SBATCH --job-name=v23-K8-fresh-eval
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v23_K8_fresh_eval_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v23_K8_fresh_eval_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments
CKPT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v23/K8_from_K1_fresh_dconv8_23k/v13_residual_step26000.pkl
[ ! -f "${CKPT}" ] && { echo "missing ${CKPT}"; exit 1; }
$PYTHON -u scripts/training/full_mamba_v20/eval_v20_rollout.py \
  --ckpt "${CKPT}" \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --residual-msg-steps 2 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-state 16 --temporal-d-conv 8 \
  --temporal-dt-rank auto --temporal-layers 2 \
  --target-steps 40 --n-samples 32 \
  --out-json /scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0526_v23_eval_K40/v23_K8_fresh_K40.json
