#!/bin/bash
#SBATCH --job-name=base-t6
#SBATCH --account=dabanin
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/DABANIN/lm8598/Weather_Global/logs/baseline_t6_%j.out
#SBATCH --error=/scratch/gpfs/DABANIN/lm8598/Weather_Global/logs/baseline_t6_%j.err

PROJECT_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
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
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export PYTHONPATH="${PROJECT_ROOT}/third_party/graphcast:${PYTHONPATH:-}"

echo "========== baseline_t6 (no mamba, target_steps=6, 2000 steps) =========="
python -u scripts/training/train_graphcast.py \
  --data-path data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --stats-dir data/graphcast/graphcast/stats \
  --val-year 2022 --train-start-year 2020 --train-end-year 2021 \
  --resolution 2.0 --mesh-size 4 --width 128 --processor-msg-steps 1 \
  --batch-size 1 --max-steps 2000 --eval-every 500 --eval-batch-size 8 \
  --checkpoint-every 500 --lr 1e-4 --weight-decay 1e-4 --seed 0 \
  --precision bf16 --input-duration 12h --target-steps 6 \
  --out-dir artifacts/checkpoints/smoke_test --run-name baseline_t6
