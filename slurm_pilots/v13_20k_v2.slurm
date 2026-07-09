#!/usr/bin/env bash
# v13 v2 = v9 architecture trained on 3yr (2020-2021 train, 2022 val) of
# paper-standard WB2-cloud data. Provides apples-to-apples baseline against
# v15 v2 (same arch, 7yr).
# Reuses v15 v2 residuals via symlink (deterministic — same data, same seed).
#
# Depends: v15-data-prep-v2 (8006416) must succeed first.

#SBATCH --job-name=v13-20k-v2
#SBATCH --account=biao
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --constraint=gpu80
#SBATCH --time=08:00:00
#SBATCH --output=/home/lm8598/Weather_Global_experiments/logs/v13_20k_v2_%j.out
#SBATCH --error=/home/lm8598/Weather_Global_experiments/logs/v13_20k_v2_%j.err

set -euo pipefail
mkdir -p /home/lm8598/Weather_Global_experiments/logs
export MPLCONFIGDIR=/tmp/mpl-weather-global
mkdir -p "${MPLCONFIGDIR}"

PYTHON=/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python
SCRIPT=/home/lm8598/Weather_Global_experiments/scripts/training/full_mamba_v13/train_mz_v13.py
cd /home/lm8598/Weather_Global_experiments

CKPT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
PREPARED_ROOT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1
RES_7YR=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v15_setup_res1_7yr_v2
RES_3YR=/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/precomputed_residuals/v13_setup_res1_3yr_v2
OUT=/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v13_v9arch_3yr_v2

# === Step 1: build 3yr setup (anchors filtered to 2020-2021 train, 2022 val) ===
echo "============================================================"
echo "Building 3yr setup at ${RES_3YR}"
echo "============================================================"
${PYTHON} - <<EOF
import os, shutil
import numpy as np
from pathlib import Path

src = Path("${RES_7YR}")
dst = Path("${RES_3YR}")

if dst.exists():
    print(f"[3yr-setup] {dst} already exists, will overwrite")
    shutil.rmtree(dst)
dst.mkdir(parents=True, exist_ok=True)
(dst / "anchors").mkdir()

# Symlink the residual data (deterministic — same residuals as 7yr setup)
res_link = dst / "residuals"
os.symlink(src / "residuals", res_link)
# Symlink metadata
os.symlink(src / "metadata.json", dst / "metadata.json")

# Filter train split to 2020-2021 only
anchor_times = np.load(src / "anchors" / "anchor_times.npy")  # shape (N_total,)
train_split = np.load(src / "anchors" / "split_train.npy")    # positions in residuals/anchors
val_split   = np.load(src / "anchors" / "split_val.npy")

# Determine which train positions correspond to 2020-2021 anchors
times_at_train = anchor_times[train_split]
years = times_at_train.astype('datetime64[Y]').astype(int) + 1970
mask = (years >= 2020) & (years <= 2021)
train_3yr = train_split[mask]
print(f"[3yr-setup] full 7yr train: {train_split.size} anchors")
print(f"[3yr-setup] filtered 3yr train (2020-2021): {train_3yr.size} anchors")
print(f"[3yr-setup] val (2022) unchanged: {val_split.size} anchors")

# Copy anchor_indices and anchor_times unchanged (the train script uses these
# to index into the residuals memmap). Save filtered split_train.
shutil.copy(src / "anchors" / "anchor_indices.npy", dst / "anchors" / "anchor_indices.npy")
shutil.copy(src / "anchors" / "anchor_times.npy",   dst / "anchors" / "anchor_times.npy")
np.save(dst / "anchors" / "split_train.npy", train_3yr)
np.save(dst / "anchors" / "split_val.npy",   val_split)

# Sanity: train year span
y_set = sorted(set((anchor_times[train_3yr].astype('datetime64[Y]').astype(int) + 1970).tolist()))
print(f"[3yr-setup] train years: {y_set}")
print(f"[3yr-setup] DONE")
EOF

# === Step 2: train v13 v2 ===
echo
echo "============================================================"
echo "Training v13 v2 (v9 arch, 3yr standard data)"
echo "============================================================"
${PYTHON} -u ${SCRIPT} \
  --prepared-root "${PREPARED_ROOT}" \
  --residual-root "${RES_3YR}" \
  --ckpt-in "${CKPT}" \
  --stats-dir /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats \
  --out-dir "${OUT}" \
  --run-name v13_20k_v2 \
  --resolution 1.0 --mesh-size 5 --width 512 \
  --residual-msg-steps 2 \
  --input-duration 12h --target-steps 1 \
  --batch-size 1 \
  --max-steps 20000 --checkpoint-every 1000 \
  --lr 1e-4 --weight-decay 1e-4 --seed 0 --precision bf16 \
  --sequential-segment-steps 32 --bptt-steps 8 \
  --temporal-location mesh_processor_interleaved \
  --temporal-hidden-size 128 \
  --temporal-d-state 16 --temporal-d-conv 4 --temporal-dt-rank auto \
  --temporal-layers 2
