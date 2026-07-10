#!/usr/bin/env bash
# Dispatcher: scan ckpt dir for new ckpts, submit eval array job for missing ones.
set -euo pipefail
CKPT_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res2
OUT_DIR=/home/lm8598/Weather_Global_experiments/results/2026-6-25-Res2Mamba
mkdir -p $OUT_DIR

# Gather missing tasks
TASK_K=()
TASK_STEP=()
for K in 2 4 8 10 14 18; do
  K_DIR=$CKPT_ROOT/K${K}_res2_fresh
  [ -d "$K_DIR" ] || continue
  for ckpt in $(ls -1 $K_DIR/v13_residual_step*.pkl 2>/dev/null | sort -V); do
    STEP=$(basename $ckpt | sed 's/v13_residual_step\([0-9]*\)\.pkl/\1/')
    OUT_JSON=$OUT_DIR/K${K}_step${STEP}_cold_bp.json
    if [ ! -f "$OUT_JSON" ]; then
      TASK_K+=($K)
      TASK_STEP+=($STEP)
    fi
  done
done

N=${#TASK_K[@]}
if [ $N -eq 0 ]; then
  echo "[dispatcher] no new ckpts to eval"
  exit 0
fi
echo "[dispatcher] submitting eval for $N ckpts:"
for i in $(seq 0 $((N-1))); do
  echo "  K=${TASK_K[$i]} step=${TASK_STEP[$i]}"
done

# Also check if there's already a queued/running eval for this set
EXISTING_PENDING=$(squeue -u lm8598 -h -o "%j" 2>/dev/null | grep -c "v22-res2-autoeval" || true)
if [ $EXISTING_PENDING -gt 0 ]; then
  echo "[dispatcher] $EXISTING_PENDING auto-eval task(s) already queued; skipping submission"
  exit 0
fi

# Build dynamic array slurm
TMPSLURM=/tmp/v22_res2_autoeval_$$.slurm
cat > $TMPSLURM <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=v22-res2-autoeval
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --array=0-$((N-1))%6
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22_res2_autoeval_t%a_%A.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22_res2_autoeval_t%a_%A.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p \$MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="\${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
cd /home/lm8598/Weather_Global_experiments_v22clean

TASK_K=(${TASK_K[@]})
TASK_STEP=(${TASK_STEP[@]})
K=\${TASK_K[\$SLURM_ARRAY_TASK_ID]}
STEP=\${TASK_STEP[\$SLURM_ARRAY_TASK_ID]}

ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
GCV1_K3=\${ROOT}/results/GCv1/K3_cascade_res2_m4_w512_mp6/ckpt_step52000.npz
CKPT=\${ROOT}/results/v22_res2/K\${K}_res2_fresh/v13_residual_step\${STEP}.pkl
OUT_JSON=$OUT_DIR/K\${K}_step\${STEP}_cold_bp.json

[ -f "\${CKPT}" ] || { echo "ERR: missing ckpt \${CKPT}"; exit 1; }
if [ -f "\${OUT_JSON}" ]; then
  echo "[skip] already done: \${OUT_JSON}"
  exit 0
fi

echo "=== K=\${K} step=\${STEP} cold_bp eval ==="

\$PYTHON -u scripts/training/full_mamba_v23/eval_v22_clean.py \\
  --ckpt \${CKPT} \\
  --ckpt-in \${GCV1_K3} \\
  --data-path \${ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \\
  --stats-dir \${ROOT}/data/graphcast/graphcast/stats \\
  --resolution 2.0 --mesh-size 4 --width 512 \\
  --baseline-msg-steps 6 --residual-msg-steps 2 \\
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \\
  --input-duration 12h \\
  --temporal-location mesh_processor_interleaved \\
  --temporal-hidden-size 128 --temporal-d-inner 128 \\
  --temporal-d-state 16 --temporal-d-conv 4 \\
  --temporal-dt-rank auto --temporal-layers 2 \\
  --eval-mode cold_bp \\
  --target-steps 40 \\
  --warmup-steps 24 \\
  --n-samples 32 \\
  --seed 0 \\
  --out-json "\${OUT_JSON}"

echo "K=\${K} step=\${STEP} cold_bp done"
EOF
sbatch $TMPSLURM
rm $TMPSLURM
echo "[dispatcher] submitted array job for $N tasks"
