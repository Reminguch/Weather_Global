#!/usr/bin/env python
"""Regenerate v22 K-scan plots from the new zero-init re-eval JSONs.

Mirrors the structure of /home/lm8598/.../2026-05-23-v22/plots/, outputs to
/home/lm8598/.../2026-06-27-v22-zero-init/plots/.

Source JSONs: /home/lm8598/Weather_Global_experiments/results/2026-6-26-v22-kscan-zero-init/
                v22_K{K}_K40_zero_init.json  for K ∈ {1,2,4,6,8,10,12,14,16,18,20,22}

Outputs:
  - plots/v22_zero_init_vs_baseline.png   — 2-panel overlay (all K)
  - plots/K{K}/4metrics_K40.png
  - plots/K{K}/per_variable_improvement_K40.png
  - plots/K{K}/per_variable_rmse_K40.png
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import xarray as xr

ROOT = Path("/home/lm8598/Weather_Global_experiments")
JSON_DIR = ROOT / "results/2026-6-26-v22-kscan-zero-init"
OUT_BASE = ROOT / "results/2026-06-27-v22-zero-init/plots"
PY = ROOT / ".conda/envs/graphcast311/bin/python"
KSCAN = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
VERSION = "v22_zero_init"
D_CONV = 4

W_VAR = {
    "10m_u_component_of_wind": 0.1, "10m_v_component_of_wind": 0.1,
    "2m_temperature": 1.0, "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1, "geopotential": 1.0,
    "temperature": 1.0, "u_component_of_wind": 1.0,
    "v_component_of_wind": 1.0, "vertical_velocity": 1.0, "specific_humidity": 1.0,
}
SIGMA_DS = xr.open_dataset(
    "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats/diffs_stddev_by_level.nc")
SIGMA = {v: float(SIGMA_DS[v].values if SIGMA_DS[v].values.ndim == 0
                  else SIGMA_DS[v].values.mean())
         for v in W_VAR if v in SIGMA_DS}


def _mse_imp(ev, k_idx):
    tb = tf = 0.0
    for v in ev["per_variable_per_step"]:
        if v not in SIGMA: continue
        w = W_VAR.get(v, 1.0); s = SIGMA[v]
        rb = ev["per_variable_per_step"][v]["rmse_baseline"][k_idx]
        rf = ev["per_variable_per_step"][v]["rmse_full"][k_idx]
        tb += w * rb**2 / s**2; tf += w * rf**2 / s**2
    return (1 - tf / tb) * 100 if tb > 0 else 0


def make_overlay():
    evs = {}
    Ks = []
    for K in KSCAN:
        p = JSON_DIR / f"v22_K{K}_K40_zero_init.json"
        if not p.exists():
            print(f"[skip] {p.name} missing"); continue
        evs[K] = json.loads(p.read_text()); Ks.append(K)
    if not Ks:
        print("no JSONs"); return
    n_lead = evs[Ks[0]]["target_steps"]
    lead_h = [6 * (i + 1) for i in range(n_lead)]
    SELECTED = [6, 24, 48, 72, 120, 168, 240]
    available = [h for h in SELECTED if h <= max(lead_h)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    cmap = cm.get_cmap("viridis")
    norm = plt.Normalize(min(Ks), max(Ks))

    ax = axes[0]
    for K in Ks:
        imps = [_mse_imp(evs[K], k) for k in range(n_lead)]
        ax.plot(lead_h, imps, "-o", ms=4, lw=1.5, color=cmap(norm(K)),
                label=f"{VERSION} K={K}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lead time (h)"); ax.set_ylabel("improvement vs baseline (%)")
    ax.set_title(f"{VERSION} (chunk=24, d_conv={D_CONV}) — paper formula, lead 6h..{max(lead_h)}h")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax = axes[1]
    cmap2 = cm.get_cmap("tab10")
    for i, lh in enumerate(available):
        k = lh // 6 - 1
        imps = [_mse_imp(evs[K], k) for K in Ks]
        ax.plot(Ks, imps, "-o", ms=5, lw=1.8, color=cmap2(i), label=f"lead {lh}h")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("K (training AR tail)"); ax.set_ylabel("improvement (%)")
    ax.set_title(f"{VERSION} K-scan"); ax.set_xticks(Ks); ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"{VERSION} (chunk=24, d_conv={D_CONV}, true zero-init) vs GraphCast — "
        f"paper formula, lead 6h..{max(lead_h)}h ({max(lead_h)//24} days)",
        fontsize=12)
    plt.tight_layout()
    out = OUT_BASE / f"{VERSION}_vs_baseline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def make_per_K():
    plot_script = ROOT / "scripts/training/full_mamba_v23/plot_v23_kscan.py"
    if not plot_script.exists():
        print(f"missing plot script {plot_script}"); return
    for K in KSCAN:
        json_path = JSON_DIR / f"v22_K{K}_K40_zero_init.json"
        if not json_path.exists():
            print(f"[skip] K={K} no JSON"); continue
        out_dir = OUT_BASE / f"K{K}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [str(PY), str(plot_script),
               str(json_path), str(out_dir), VERSION, str(D_CONV)]
        print(f"\n=== K={K} ===\n  $ {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  STDERR: {res.stderr[-400:]}")
        else:
            print(f"  OK → {out_dir}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or "overlay" in args:
        make_overlay()
    if not args or "perK" in args:
        make_per_K()
