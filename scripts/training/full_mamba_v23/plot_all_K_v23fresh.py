#!/usr/bin/env python
"""Generate per-K detail plots for v23 fresh K-scan (mirror of v22 plots/K{K}/).

For each K in {1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}:
  → 4metrics_K40.png, per_variable_improvement_K40.png, per_variable_rmse_K40.png
    (via plot_v23_kscan.py)
  → extreme_records_K40.png (via plot_extreme_records_v23.py)

Output: /scratch/.../2026-05-30-v23/plots/K{K}_fresh/
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/lm8598/Weather_Global_experiments")
EVAL_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0526_v23_eval_K40")
EXTREME_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0526_v23_extreme")
PLOT_BASE = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/"
                  "results/2026-05-30-v23/plots")
PY = "/home/lm8598/Weather_Global_experiments/.conda/envs/graphcast311/bin/python"

KS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]


def run(cmd: list[str]) -> bool:
    print(f"  $ {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  STDERR: {res.stderr[-500:]}")
        return False
    return True


def main() -> None:
    for K in KS:
        json_path = EVAL_DIR / f"v23_K{K}_fresh_K40.json"
        extreme_path = EXTREME_DIR / f"v23_K{K}_fresh_extreme_records_K40_2022.nc"
        out_dir = PLOT_BASE / f"K{K}_fresh"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== K={K} ===")

        # 1. Per-K 4metrics + per-variable plots (via plot_v23_kscan.py)
        if json_path.exists():
            run([PY, str(ROOT / "scripts/training/full_mamba_v23/plot_v23_kscan.py"),
                 str(json_path), str(out_dir), "v23", "8"])
        else:
            print(f"  SKIP eval plots: {json_path} not yet ready")

        # 2. Extreme records plot — script writes to its OUT_BASE / K{K}_{variant}/.
        # We then move the result into our plots/K{K}_fresh/ dir.
        if extreme_path.exists():
            ok = run([PY, str(ROOT / "scripts/training/full_mamba_v23/plot_extreme_records_v23.py"),
                      str(K), "fresh"])
            if ok:
                src = Path("/home/lm8598/Weather_Global_experiments/results/"
                           f"2026-05-26-v23/plots/K{K}_fresh/extreme_records_K40.png")
                dst = out_dir / "extreme_records_K40.png"
                if src.exists():
                    src.rename(dst)
                    print(f"  moved {src} -> {dst}")
        else:
            print(f"  SKIP extreme plot: {extreme_path} not yet ready")


if __name__ == "__main__":
    main()
