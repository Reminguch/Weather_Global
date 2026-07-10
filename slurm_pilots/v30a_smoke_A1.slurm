#!/usr/bin/env bash
# v30a SMOKE A1 — verify force_init_target_encoder populates params tree.
#
# Goal: with --force-init-target-encoder ON, the target_encoder MLP params land
# in the Haiku tree at init time (via a dummy event_batch threaded into init),
# even though lambda_con=0 and event_batch=None at every training step.
#
# Predictions and loss should STILL equal v22 byte-identically (because
# adapter_scale=0). The only observable difference vs A0 should be:
#   - "[v30] dummy event_batch threaded into init to populate
#      temporal_event_target_encoder/* params." log appears
#   - residual_params count higher by ~ hidden*flat + flat*proj parameter counts
#   - PHASE 1 trainable count includes target_encoder leaves
#
#SBATCH --job-name=v30a-A1
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v30a_A1_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v30a_A1_%j.err
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

echo "=== runtime guard: verify v30 file path + Step-1 patches ==="
${PYTHON} - <<'PY'
import inspect, sys
sys.path.insert(0, ".")
import scripts.training.full_mamba_v30.event_adapter_head as m
from scripts.training.full_mamba_v30.event_adapter_head import GCResidualWithEventAdapter
print(f"[v30 file] {m.__file__}")
src = inspect.getsource(GCResidualWithEventAdapter)
expected = [
    "temporal_event_damp",
    "effective_adapter",
    "force_init_target_encoder",
    "self._adapter_scale == 0.0",
    "self._event_patch_size",
]
missing = [pat for pat in expected if pat not in src]
if missing:
    print(f"FATAL: missing patches in event_adapter_head.py: {missing}")
    sys.exit(2)
print("[v30 guard] all Step-1 patches present.")
PY

echo "=== v30a A1: adapter_scale=0, lambda_con=0, force_init_target_encoder=TRUE ==="

${PYTHON} -u scripts/training/full_mamba_v20/train_mz_v20.py \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RESIDUAL_ROOT}" \
  --ckpt-in "${GCV1_K3}" \
  --stats-dir ${ROOT}/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name A1_scale0_force_init \
  --resolution 2.0 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 3 --checkpoint-every 1 \
  --lr 3e-6 --weight-decay 0.0 --seed 30 --precision bf16 \
  --grad-clip 1.0 --warmup-steps 0 \
  --sequential-segment-steps 96 --bptt-steps 24 \
  --ar-tail-K 18 \
  --feedback-mode baseline \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-inner 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2 \
  --event-adapter-head \
  --init-from-residual-ckpt "${INIT_CKPT}" \
  --gate-phase 1 \
  --adapter-scale 0.0 \
  --lambda-con 0.0 \
  --lambda-delta-reg 0.0 \
  --event-trunk-dim 64 \
  --event-proj-dim 64 \
  --event-gate-bias-init -2.0 \
  --damp-bias-init -4.0 \
  --event-delta-cap 0.5 \
  --event-head-lr 1e-4 \
  --event-target-encoder-lr 1e-4 \
  --force-init-target-encoder

echo "=== v30a A1 done. Sanity-check stdout for: ==="
echo "  - '[v30] dummy event_batch threaded' log present"
echo "  - 'kept at fresh init' includes temporal_event_target_encoder/*"
echo "  - PHASE 1 trainable count INCREASED vs A0 (extra target_encoder leaves)"
echo "  - step 0/1 loss MUST EQUAL A0's step 0/1 loss byte-identical"
