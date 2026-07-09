#!/bin/bash
# Baseline fine-tune at K=4 (24h rollout), starting from K=1 ckpt_step23000.
# BPTT over 4 steps per sample ~ 4x slower than K=1 training; budget 5000
# extra FT steps.

#SBATCH --job-name=bfix-r4m3-ft-t4
#SBATCH --account=dabanin
#SBATCH --qos=gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/DABANIN/lm8598/Weather_Global/lagrangian_lab/logs/bfix_r4_m3_in32_t4_ft_%j.out
#SBATCH --error=/scratch/gpfs/DABANIN/lm8598/Weather_Global/lagrangian_lab/logs/bfix_r4_m3_in32_t4_ft_%j.err

PROJECT_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/lagrangian_lab
DATA_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
cd "${PROJECT_ROOT}"

module purge 2>/dev/null || true
module load anaconda3/2025.6 2>/dev/null || true
if [ -f "/usr/licensed/anaconda3/2025.6/etc/profile.d/conda.sh" ]; then
  source /usr/licensed/anaconda3/2025.6/etc/profile.d/conda.sh
else
  eval "$(conda shell.bash hook)" || true
fi
conda activate /home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311

set -euo pipefail
mkdir -p logs artifacts/checkpoints/long_iid_mamba
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export PYTHONPATH="${PROJECT_ROOT}/third_party/graphcast:${PYTHONPATH:-}"

echo "========== baseline fine-tune res=4 mesh=3 in=32 t=4 (from step23000) =========="
python -u scripts/training/train_graphcast.py \
  --data-path "${DATA_ROOT}/data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr" \
  --stats-dir "${DATA_ROOT}/data/graphcast/graphcast/stats" \
  --ckpt-in "${PROJECT_ROOT}/artifacts/checkpoints/long_iid_mamba/bfix_r4_m3_in32_t1/ckpt_step23000.npz" \
  --resume-step 23000 \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 4.0 --mesh-size 3 --width 128 --processor-msg-steps 1 \
  --batch-size 1 --max-steps 28000 --eval-every 1000 --eval-batch-size 8 \
  --checkpoint-every 1000 --lr 5e-5 --weight-decay 1e-4 --seed 0 \
  --precision bf16 --input-duration 192h --target-steps 4 \
  --temporal-backbone none \
  --out-dir artifacts/checkpoints/long_iid_mamba \
  --run-name bfix_r4_m3_in32_t4_ft
