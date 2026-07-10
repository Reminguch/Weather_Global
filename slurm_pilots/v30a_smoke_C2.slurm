#!/usr/bin/env bash
# v30a SMOKE C2 — train_step-level JIT smoke with InfoNCE at ONE anchor.
#
# Goals (per user's C2 spec):
#   1. JIT compiles when event_batch is threaded through BPTT at contrastive
#      anchor only (last anchor by default). Other anchors call the plain
#      _one_ar_step path unchanged.
#   2. Training runs 3 steps without NaN.
#   3. aux_L_con is finite at every step (printed in log line).
#   4. grad_norm > 0 (event_trunk/proj/encoder receive gradients).
#   5. effective_adapter_rms would be 0 (adapter_scale=0) — checked via
#      checkpoint diff on event_gate/delta/damp params (should be unchanged).
#   6. base loss magnitude comparable to A0/A1 (~6 range).
#
#SBATCH --job-name=v30a-C2
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v30a_C2_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v30a_C2_%j.err
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
OUT=${ROOT}/results/v30a_smoke

mkdir -p ${OUT}
[ -f "${INIT_CKPT}" ] || { echo "ERR: missing K=18 init ckpt"; exit 1; }

echo "=== runtime guard: verify v30 patches + C0 patches ==="
${PYTHON} - <<'PY'
import inspect, sys
sys.path.insert(0, ".")
import scripts.training.full_mamba_v30.event_adapter_head as m
from scripts.training.full_mamba_v30.event_adapter_head import (
    GCResidualWithEventAdapter, info_nce_loss)
print(f"[v30 file] {m.__file__}")
expected_class = ["temporal_event_damp", "effective_adapter",
                  "force_init_target_encoder", "self._adapter_scale == 0.0",
                  "self._event_patch_size", "has_negs", "z_neg",
                  "_aux_scalar"]
src_cls = inspect.getsource(GCResidualWithEventAdapter)
missing = [pat for pat in expected_class if pat not in src_cls]
if missing:
    print(f"FATAL: missing in head: {missing}"); sys.exit(2)
sig = inspect.signature(info_nce_loss)
if "z_neg" not in sig.parameters:
    print(f"FATAL: info_nce_loss missing z_neg param"); sys.exit(2)
# Check train script has C2 patches
train_src = open("scripts/training/full_mamba_v20/train_mz_v20.py").read()
for pat in ["_one_ar_step_with_event", "_use_contrastive", "_c_anchor",
            "event_batch_dummy", "contrastive_anchor_idx", "aux_lcon_captured"]:
    if pat not in train_src:
        print(f"FATAL: train_mz_v20 missing {pat}"); sys.exit(2)
print("[v30 guard] all Step-1 + C0 + C2 patches present.")
PY

echo "=== v30a C2: 3 train steps with event_batch at anchor 23 (last) ==="

${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${GCV1_K3}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name C2_contrastive_smoke \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 3 --checkpoint-every 1 \
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
  --lambda-con 0.01 \
  --lambda-delta-reg 0.0 \
  --tau-contrastive 0.1 \
  --event-patch-size 16 \
  --event-trunk-dim 64 --event-proj-dim 64 \
  --event-gate-bias-init -2.0 --damp-bias-init -4.0 --event-delta-cap 0.5 \
  --event-head-lr 1e-4 \
  --event-target-encoder-lr 1e-4 \
  --contrastive-anchor-idx 23 \
  --n-neg 4 \
  --event-dummy

echo "=== C2 sanity checklist (from stdout lines above): ==="
echo "  - PHASE 1 trainable count > 0"
echo "  - step 1/2/3 loss ~ 6 (A0/A1 baseline range)"
echo "  - step 1/2/3 aux_L_con finite (~log(5)=1.61, elevated by tau=0.1)"
echo "  - grad_norm > 0 (event params + Mamba receive gradients)"
