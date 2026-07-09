#!/usr/bin/env bash
# First production gcmamba_v1 run: K=1 truth-fed, Mamba params matched to v22.
# Cleanest internal-vs-external Mamba comparison vs v22 (res_mamba style):
#   - same backbone (DeepMind GC small, res=1.0, mesh=5, width=512, mp=16)
#   - same Mamba size (hidden=128, d_inner=128, d_state=16, d_conv=4, layers=2)
#   - same training meta (bptt=24, K=1 truth-fed, lr=1e-4, wd=1e-4, seed=22,
#     warmup=200, grad_clip=1.0, segment=96)
#   - architecture difference: Mamba INTERLEAVED INSIDE GC processor
#     (vs v22 = external residual GC + Mamba sitting on frozen baseline)
# memory_mode=conservative: Mamba-only param partition (already on),
# no remat (non-remat smoke was fast: 1.7s/step at bptt=8 -> ~5s/step at
# bptt=24). If HBM is tight at bptt=24, fall back to memory_mode=optimal.
#SBATCH --job-name=gcm-v1-run01-v22m-K1
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=24:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/gcm_v1_run01_v22m_K1_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/gcm_v1_run01_v22m_K1_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments

OUT_DIR=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/gcmamba_v1
mkdir -p ${OUT_DIR}

$PYTHON -u scripts/training/gcmamba_v1/train_gcmamba_v1.py \
  --out-dir ${OUT_DIR} \
  --run-name gcmamba_res1_mp16_bptt24_tgt1_di128_ds16_L1_yr2015-2021_seed22_optimal \
  --resolution 1.0 --mesh-size 5 --width 512 --processor-msg-steps 16 \
  --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 --temporal-d-state 16 \
  --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 1 \
  --temporal-stateful \
  --bptt-steps 24 --target-steps 1 --len-segment 96 \
  --batch-size 1 \
  --lr 1e-4 --weight-decay 1e-4 --grad-clip 1.0 \
  --warmup-steps 200 \
  --max-steps 10000 --eval-every 500 --checkpoint-every 1000 \
  --train-start-year 2015 --train-end-year 2021 \
  --val-year 2022 \
  --seed 22 \
  --memory-mode optimal
echo "gcmamba_v1 run01 (v22-matched Mamba, K=1, optimal remat) done"
