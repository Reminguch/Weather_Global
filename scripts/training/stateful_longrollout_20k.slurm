#!/bin/bash
#SBATCH --job-name=sf-t10
#SBATCH --account=dabanin
#SBATCH --qos=gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gpfs/DABANIN/lm8598/Weather_Global/logs/stateful_t10_%j.out
#SBATCH --error=/scratch/gpfs/DABANIN/lm8598/Weather_Global/logs/stateful_t10_%j.err

PROJECT_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global
cd "${PROJECT_ROOT}"

# Load conda
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
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export PYTHONPATH="${PROJECT_ROOT}/third_party/graphcast:${PYTHONPATH:-}"

DATA_PATH="data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr"
STATS_DIR="data/graphcast/graphcast/stats"
BASE_OUT="artifacts/checkpoints/graphcast_2020_2021_val2022"

echo "========== Stateful Mamba target_steps=10, input_steps=2 =========="
python -u scripts/training/train_graphcast.py \
  --data-path "${DATA_PATH}" \
  --stats-dir "${STATS_DIR}" \
  --val-year 2022 \
  --train-start-year 2020 \
  --train-end-year 2021 \
  --resolution 2.0 \
  --mesh-size 4 \
  --width 128 \
  --processor-msg-steps 1 \
  --batch-size 1 \
  --max-steps 20000 \
  --eval-every 2000 \
  --eval-batch-size 4 \
  --checkpoint-every 2000 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --seed 0 \
  --precision bf16 \
  --input-duration 12h \
  --target-steps 10 \
  --temporal-backbone mamba \
  --temporal-stateful \
  --temporal-location mesh_post_encoder \
  --out-dir "${BASE_OUT}" \
  --run-name "stateful_h2_t10_20k"

echo "========== Done =========="
