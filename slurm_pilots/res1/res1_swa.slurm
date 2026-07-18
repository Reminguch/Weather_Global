#!/usr/bin/env bash
# ============================================================================
# res=1 SWA (build + eval). Third of the matched trio. POST-PROCESSING step:
# after res1_closed_loop.slurm has produced step-6000..16000 checkpoints, this
#   1. builds a uniform SWA average of those 6 ckpts (build_swa_generic.py), then
#   2. evaluates the SWA ckpt (eval_v22_clean.py, cold_full, 32 inits, 2022).
# NOTE: "val loss" is computed HERE (post-hoc), not during training — the trainer
# logs only training loss; validation/skill is measured afterward by this eval.
# Point --run-glob at the closed-loop run you want to average.
# ============================================================================
#SBATCH --job-name=res1-swa
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=03:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/res1_swa_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/res1_swa_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_v22clean_frozen        # frozen-clean tree; relative script path

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
DM_CKPT="${ROOT}/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
CLOSED_OUT=${ROOT}/results/res1_sync/closed_loop_K${K:-20}        # from res1_closed_loop.slurm
SWA_NAME=v13_residual_SWA_step6k-16k.pkl
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/res1_sync_swa
mkdir -p ${OUT_DIR}

echo "=== 1) build SWA(6k-16k) from closed-loop K${K:-20} ckpts ==="
${PYTHON} -u scripts/training/full_mamba_v23/build_swa_generic.py \
  --run-glob "${CLOSED_OUT}/K{K}_res1_closed_loop" \
  --ks ${K:-20} --steps 6000,8000,10000,12000,14000,16000 \
  --out-name ${SWA_NAME}

SWA_CKPT=${CLOSED_OUT}/K${K:-20}_res1_closed_loop/${SWA_NAME}
# build_swa exits non-zero + deletes any stale SWA on failure; double-check the file
# was actually produced this run before evaluating (never eval a leftover stale SWA).
[ -f "${SWA_CKPT}" ] || { echo "ERR: SWA ckpt was not produced this run: ${SWA_CKPT}"; exit 1; }
echo "=== 2) eval SWA ckpt (cold_full, 32 inits, 2022) ==="
${PYTHON} -u scripts/training/full_mamba_v23/eval_v22_clean.py \
  --ckpt "${SWA_CKPT}" --ckpt-in "${DM_CKPT}" \
  --data-path ${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 --input-duration 12h \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto --temporal-layers 2 \
  --eval-mode cold_full --residual-state-init zero \
  --target-steps 40 --warmup-steps 24 --n-samples 32 --seed 0 \
  --out-json ${OUT_DIR}/res1_K${K:-20}_SWA6k16k_cold_full.json
echo "=== res=1 SWA build+eval done ==="
