#!/usr/bin/env python
"""2-panel K-scan overlay plot for a single version (v20, v22, or v23).

Left  : MSE-improvement curve vs lead time, all K colored.
Right : MSE-improvement vs K, for several selected leads.

Both panels use GraphCast paper formula with diffs_stddev_by_level normalizer.

Usage:
  python plot_kscan_overlay.py <version=v20|v22|v23> [variant=20k|23k for v23]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

W_VAR = {
    "10m_u_component_of_wind": 0.1, "10m_v_component_of_wind": 0.1,
    "2m_temperature": 1.0, "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1, "geopotential": 1.0,
    "temperature": 1.0, "u_component_of_wind": 1.0,
    "v_component_of_wind": 1.0, "vertical_velocity": 1.0, "specific_humidity": 1.0,
}
ds = xr.open_dataset("/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats/diffs_stddev_by_level.nc")
SIGMA = {v: float(ds[v].values if ds[v].values.ndim == 0 else ds[v].values.mean())
         for v in W_VAR if v in ds}


def mse_imp(ev, k_idx):
    tb = tf = 0.0
    for v in ev["per_variable_per_step"]:
        if v not in SIGMA:
            continue
        w = W_VAR.get(v, 1.0)
        s = SIGMA[v]
        rb = ev["per_variable_per_step"][v]["rmse_baseline"][k_idx]
        rf = ev["per_variable_per_step"][v]["rmse_full"][k_idx]
        tb += w * rb**2 / s**2
        tf += w * rf**2 / s**2
    return (1 - tf / tb) * 100 if tb > 0 else 0


# Per-version JSON paths
PATHS = {
    "v20": {
        "json_dir": Path("/home/lm8598/Weather_Global_experiments/results/2026-05-20-v20/eval_jsons"),
        "json_pattern": "v20_K{K}.json",
        "Ks": [1, 4, 6, 8, 10, 12, 14],
        "out_dir": Path("/home/lm8598/Weather_Global_experiments/results/2026-05-20-v20/plots"),
        "out_name": "v20_vs_baseline.png",
        "d_conv": 4,
    },
    "v22": {
        "json_dir": Path("/home/lm8598/Weather_Global_experiments/results/2026-05-23-v22/eval_jsons"),
        "json_pattern": "v22_K{K}_K40.json",
        "Ks": [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        "out_dir": Path("/home/lm8598/Weather_Global_experiments/results/2026-05-23-v22/plots"),
        "out_name": "v22_vs_baseline.png",
        "d_conv": 4,
    },
    "v23_20k": {
        "json_dir": Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0526_v23_eval_K40"),
        "json_pattern": "v23_K{K}_20k_K40.json",
        "Ks": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        "out_dir": Path("/home/lm8598/Weather_Global_experiments/results/2026-05-26-v23/plots"),
        "out_name": "v23_20k_vs_baseline.png",
        "d_conv": 8,
    },
    "v23_23k": {
        "json_dir": Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0526_v23_eval_K40"),
        "json_pattern": "v23_K{K}_23k_K40.json",
        "Ks": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        "out_dir": Path("/home/lm8598/Weather_Global_experiments/results/2026-05-26-v23/plots"),
        "out_name": "v23_23k_vs_baseline.png",
        "d_conv": 8,
    },
}


def plot_one(version_key: str):
    cfg = PATHS[version_key]
    Ks_present = []
    evs = {}
    for K in cfg["Ks"]:
        p = cfg["json_dir"] / cfg["json_pattern"].format(K=K)
        if p.exists():
            evs[K] = json.loads(p.read_text())
            Ks_present.append(K)

    if not Ks_present:
        print(f"no eval JSONs found for {version_key}")
        return

    # Get n_lead from any one ev (assume same across K)
    n_lead = evs[Ks_present[0]]["target_steps"]
    lead_h = [6 * (i + 1) for i in range(n_lead)]
    # Selected leads for right panel (in hours), filter to those available
    SELECTED_LEAD_H = [6, 24, 48, 72, 120, 168, 240]
    available_lead_h = [h for h in SELECTED_LEAD_H if h <= max(lead_h)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # Color per K (viridis colormap)
    cmap = cm.get_cmap("viridis")
    norm = plt.Normalize(min(Ks_present), max(Ks_present))

    # LEFT panel: improvement % vs lead, all K curves
    ax = axes[0]
    for K in Ks_present:
        imps = [mse_imp(evs[K], k_idx) for k_idx in range(n_lead)]
        ax.plot(lead_h, imps, "-o", ms=4, lw=1.5, color=cmap(norm(K)),
                label=f"{version_key} K={K}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("improvement vs baseline (%)")
    ax.set_title(f"{version_key} (chunk=24, d_conv={cfg['d_conv']}) vs GraphCast baseline — lead 6h..{max(lead_h)}h")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    # RIGHT panel: improvement % vs K, for several leads
    ax = axes[1]
    cmap2 = cm.get_cmap("tab10")
    for i, lh in enumerate(available_lead_h):
        k_idx = lh // 6 - 1
        imps = [mse_imp(evs[K], k_idx) for K in Ks_present]
        ax.plot(Ks_present, imps, "-o", ms=5, lw=1.8, color=cmap2(i), label=f"lead {lh}h")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("K (training AR tail)")
    ax.set_ylabel("improvement (%)")
    ax.set_title(f"{version_key} K-scan")
    ax.set_xticks(Ks_present)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"{version_key} (chunk=24, d_conv={cfg['d_conv']}) vs GraphCast baseline — paper formula, "
        f"lead 6h..{max(lead_h)}h ({max(lead_h)//24} days)",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = cfg["out_dir"] / cfg["out_name"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


def main():
    if len(sys.argv) < 2:
        # default: regenerate all available
        for key in ["v20", "v22", "v23_20k", "v23_23k"]:
            plot_one(key)
    else:
        plot_one(sys.argv[1])


if __name__ == "__main__":
    main()
