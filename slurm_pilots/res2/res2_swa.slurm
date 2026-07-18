#!/usr/bin/env bash
# ============================================================================
# res=2 SWA (build + eval). Third of the matched trio. POST-PROCESSING:
# after res2_closed_loop.slurm produces step-1000/1500/2000 ckpts, this
#   1. builds a uniform SWA average of those (build_swa_generic.py), then
#   2. evaluates it (eval_v22_clean.py, cold_full, res=2). "val loss" is here,
#      not in the training loop. K is a parameter (${K:-18}).
# ============================================================================
#SBATCH --job-name=res2-swa
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --constraint=gpu80
#SBATCH --time=03:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/res2_swa_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/res2_swa_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_v22clean_frozen        # frozen-clean tree; relative script path

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
DATA=${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr   # --resolution 2.0 downsamples
CLOSED_OUT=${ROOT}/results/res2_sync/closed_loop_K${K:-18}       # from res2_closed_loop.slurm
SWA_NAME=v13_residual_SWA_step1k-2k.pkl
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/res2_sync_swa
mkdir -p ${OUT_DIR}

echo "=== 1) build SWA(1k-2k) from closed-loop K${K:-18} ckpts ==="
${PYTHON} -u scripts/training/full_mamba_v23/build_swa_generic.py \
  --run-glob "${CLOSED_OUT}/K{K}_res2_closed_loop" \
  --ks ${K:-18} --steps 1000,1500,2000 \
  --out-name ${SWA_NAME}

SWA_CKPT=${CLOSED_OUT}/K${K:-18}_res2_closed_loop/${SWA_NAME}
# build_swa exits non-zero + deletes any stale SWA on failure; double-check the file
# was actually produced this run before evaluating (never eval a leftover stale SWA).
[ -f "${SWA_CKPT}" ] || { echo "ERR: SWA ckpt was not produced this run: ${SWA_CKPT}"; exit 1; }
echo "=== 2) eval SWA ckpt (cold_full, res=2, 32 inits, 2022) ==="
${PYTHON} -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt "${SWA_CKPT}" --ckpt-in "${GCV1_K3}" \
  --data-path ${DATA} \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2 \
  --eval-mode cold_full --residual-state-init zero \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
  --out-json ${OUT_DIR}/res2_K${K:-18}_SWA1k2k_cold_full.json
echo "=== res=2 SWA build+eval done ==="
