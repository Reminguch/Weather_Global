#!/usr/bin/env bash
# v30a SMOKE B0-v3 — HIGH-SIGNAL probe with full diagnostics.
#
# B0 v2 (500 steps) got stuck at aux_L_con ≈ log(5) with grad_norm ~1e-3.
# Diagnosis: effective LR (lambda_con=0.03 × lr=1e-4 = 3e-6) too weak to
# escape the uniform-softmax random init.
#
# B0-v3 changes ONLY learning-signal knobs, NOT sampler:
#   lambda_con              0.03 → 1.0    (33x)
#   event_head_lr           1e-4 → 1e-3   (10x)
#   event_target_enc_lr     1e-4 → 1e-3   (10x)
#   steps                    500 → 300
# Everything else (sampler rules, T_win=1, patch=16, Nneg=4,
# min_time_gap_days=7) held constant. Adapter still off (adapter_scale=0),
# base still frozen (Phase 1).
#
# Success criteria (any of):
#   aux_L_con drops below 1.0  (was stuck at 1.61)
#   aux_con_gap becomes clearly positive
#   aux_con_top1 > 0.5
#
# Failure diagnosis via new aux:
#   z_pred_std near 0        → embedding collapse
#   pos_sim ≈ neg_sim ≈ 0    → logits all near uniform
#   pos_sim high, gap 0      → time alignment likely wrong
#
#SBATCH --job-name=v30a-B0v3
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v30a_B0v3_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v30a_B0v3_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p $MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
INIT_CKPT=${ROOT}/results/v22_res2/K18_res2_fresh/v13_residual_step20000.pkl
PREPARED_ROOT=/scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/dataset/prepared_stream/res2
RESIDUAL_ROOT=${ROOT}/data/graphcast/graphcast/dataset/precomputed_residuals/v22_res2_GCv1K3_synthetic
OUT=${ROOT}/results/v30a_B0

mkdir -p ${OUT}
[ -f "${INIT_CKPT}" ] || { echo "ERR: missing K=18 init ckpt"; exit 1; }

echo "=== runtime guard: v30 + B0 patches + diagnostics ==="
${PYTHON} - <<'PY'
import inspect, sys
sys.path.insert(0, ".")
from scripts.training.full_mamba_v30 import (
    GCResidualWithEventAdapter, PerStepEventSampler)
src_cls = inspect.getsource(GCResidualWithEventAdapter)
for pat in ["aux_con_neg_sim_max", "aux_z_pred_std", "aux_z_truth_std"]:
    if pat not in src_cls:
        print(f"FATAL: head missing {pat}"); sys.exit(2)
train_src = open("scripts/training/full_mamba_v20/train_mz_v20.py").read()
for pat in ["_AUX_KEYS", "aux_captured", "v30 sanity"]:
    if pat not in train_src:
        print(f"FATAL: train_mz_v20 missing {pat}"); sys.exit(2)
print("[v30 guard] diagnostics + B0-v3 wiring present.")
PY

echo "=== v30a B0-v3: HIGH-SIGNAL probe (lambda_con=1.0, event_lr=1e-3, 300 steps) ==="

${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${GCV1_K3}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name B0v3_high_signal_probe \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 300 --checkpoint-every 100 \
  --lr 3e-6 --weight-decay 0.0 --seed 30 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 0 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 18 \
  --feedback-mode baseline \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2 \
  --event-adapter-head \
  --init-from-residual-ckpt "${INIT_CKPT}" \
  --gate-phase 1 \
  --adapter-scale 0.0 \
  --lambda-con 1.0 \
  --lambda-delta-reg 0.0 \
  --tau-contrastive 0.1 \
  --event-patch-size 16 \
  --event-t-win 1 \
  --event-c-vars 4 \
  --event-trunk-dim 64 --event-proj-dim 64 \
  --event-gate-bias-init -2.0 --damp-bias-init -4.0 --event-delta-cap 0.5 \
  --event-head-lr 1e-3 \
  --event-target-encoder-lr 1e-3 \
  --contrastive-anchor-idx 23 \
  --n-neg 4 \
  --event-min-time-gap-days 7

echo "=== B0-v3 done ==="
