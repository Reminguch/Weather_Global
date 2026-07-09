#!/usr/bin/env bash
# Recovery: K=2..20 jobs all TIMEOUT at ~step 7800. Resume from latest saved
# ckpt (step 7000 / 5000 / 3000 depending on K) → train to step 20000 in ONE
# long job (32h slurm time, max 17k steps × 5.3s = 25h).
#
# Critical fix: corrected nested run_name path in --resume-from. Previous
# extend_template missed the nested dir layer (out_dir/run_name/ckpt).
#
# Output dir: /scratch/.../v22closed/K{K}_sg_fresh_20k/ (NEW dir, not
# overwriting K{K}_sg_fresh_10k from the timed-out original).

set -euo pipefail

RESULTS_ROOT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed"
SLURM_OUT_DIR="$(dirname "$0")"

declare -A LATEST_STEP
LATEST_STEP[2]=7000
LATEST_STEP[4]=7000
LATEST_STEP[6]=7000
LATEST_STEP[8]=7000
LATEST_STEP[10]=7000
LATEST_STEP[12]=7000
LATEST_STEP[14]=7000
LATEST_STEP[16]=5000
LATEST_STEP[18]=3000
LATEST_STEP[20]=3000

for K in 2 4 6 8 10 12 14 16 18 20; do
  STEP=${LATEST_STEP[$K]}
  RESUME_CKPT="${RESULTS_ROOT}/K${K}_sg_fresh_10k/v22closed_sg_K${K}_fresh_10k/v13_residual_step${STEP}.pkl"
  if [ ! -f "$RESUME_CKPT" ]; then
    echo "MISSING ckpt for K=$K at step $STEP — skip"
    continue
  fi

  SLURM_FILE="${SLURM_OUT_DIR}/v22closed_sg_K${K}_recovery_to20k.slurm"
  START_STEP=$((STEP + 1))

  cat > "$SLURM_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=v22closed-sg-K${K}-recov20k
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=32:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v22closed_sg_K${K}_recov20k_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v22closed_sg_K${K}_recov20k_%j.err
set -euo pipefail
export MPLCONFIGDIR=/tmp/mpl-weather-global; mkdir -p \$MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_CPP_MIN_LOG_LEVEL=1
PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v22closed/train_mz_v22closed.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RESIDUAL_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2

\${PYTHON} -u \${SCRIPT} \\
  --prepared-root "\${PREPARED_ROOT}" \\
  --residual-root "\${RESIDUAL_ROOT}" \\
  --ckpt-in "\${CKPT}" \\
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \\
  --out-dir ${RESULTS_ROOT} \\
  --run-name K${K}_sg_fresh_20k \\
  --resume-from "${RESUME_CKPT}" \\
  --start-step ${START_STEP} \\
  --resolution 1.0 --mesh-size 5 --width 512 \\
  --baseline-msg-steps 16 \\
  --residual-msg-steps 2 \\
  --input-duration 12h --target-steps 1 \\
  --batch-size 1 \\
  --max-steps 20000 --checkpoint-every 1000 \\
  --lr 1e-4 --weight-decay 1e-4 --seed 22 --precision bf16 \\
  --grad-clip 1.0 --warmup-steps 0 \\
  --sequential-segment-steps 96 --bptt-steps 24 \\
  --ar-tail-K ${K} \\
  --feedback-stop-gradient \\
  --temporal-location mesh_processor_interleaved \\
  --temporal-hidden-size 128 \\
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \\
  --temporal-layers 2
EOF

  JID=$(sbatch "$SLURM_FILE" | awk '{print $4}')
  echo "K=${K}: submit recovery (resume from step ${STEP} → 20000) → job ${JID}"
done
