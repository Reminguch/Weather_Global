#!/usr/bin/env python
"""K-scan summary plot: aggregate MSE-imp vs K, at multiple key lead times.

Aggregate uses GraphCast paper formula with diffs_stddev_by_level normalizer.

Outputs ONE PNG showing all K's together, with multiple lead times overlaid.
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

# Reuse W_VAR + SIGMA_V from plot_v23_kscan (paste-import to keep this standalone)
W_VAR = {
    "10m_u_component_of_wind": 0.1, "10m_v_component_of_wind": 0.1,
    "2m_temperature": 1.0, "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1, "geopotential": 1.0,
    "temperature": 1.0, "u_component_of_wind": 1.0,
    "v_component_of_wind": 1.0, "vertical_velocity": 1.0, "specific_humidity": 1.0,
}

def _load_sigma(path="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats/diffs_stddev_by_level.nc"):
    ds = xr.open_dataset(path)
    out = {}
    for v in W_VAR:
        if v not in ds: continue
        arr = ds[v].values
        out[v] = float(arr) if arr.ndim == 0 else float(arr.mean())
    return out

SIGMA_V = _load_sigma()


def mse_imp(ev, k_idx):
    variables = list(ev["per_variable_per_step"].keys())
    tb = tf = 0.0
    for v in variables:
        w = W_VAR.get(v, 1.0)
        s = SIGMA_V.get(v, None)
        if s is None or s <= 0: continue
        rb = ev["per_variable_per_step"][v]["rmse_baseline"][k_idx]
        rf = ev["per_variable_per_step"][v]["rmse_full"][k_idx]
        tb += w * rb**2 / s**2
        tf += w * rf**2 / s**2
    return (1 - tf/tb) * 100 if tb > 0 else 0


def gather(json_dir: Path, pattern: str, Ks: list[int]) -> dict:
    """Return {K: ev_dict}."""
    out = {}
    for K in Ks:
        p = json_dir / pattern.format(K=K)
        if p.exists():
            out[K] = json.loads(p.read_text())
    return out


def main():
    V22_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-05-23-v22/eval_jsons")
    V23_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0526_v23_eval_K40")
    OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-05-26-v23/plots")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v22 = gather(V22_DIR, "v22_K{K}_K40.json", [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    v23_20k = gather(V23_DIR, "v23_K{K}_20k_K40.json", [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    v23_23k = gather(V23_DIR, "v23_K{K}_23k_K40.json", [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])

    # Pick representative lead times: 24h, 96h (4d), 168h (7d), 240h (10d)
    leads = [4, 16, 28, 40]  # idx; lead_h = idx*6
    fig, axes = plt.subplots(1, len(leads), figsize=(5 * len(leads), 5), sharey=True)

    for ax, lead_idx in zip(axes, leads):
        lead_h = lead_idx * 6
        for name, runs, color, marker in [
            ("v22 (d_conv=4)", v22, "C3", "o"),
            ("v23 20k (d_conv=8)", v23_20k, "C0", "s"),
            ("v23 23k (d_conv=8)", v23_23k, "C2", "D"),
        ]:
            if not runs:
                continue
            Ks = sorted(runs.keys())
            imps = [mse_imp(runs[K], lead_idx - 1) for K in Ks]
            ax.plot(Ks, imps, marker=marker, lw=1.8, ms=6, color=color, label=f"{name}, n={len(Ks)}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("K (training AR-tail length)")
        ax.set_ylabel("aggregate MSE improvement %" if lead_idx == leads[0] else "")
        ax.set_title(f"lead = {lead_h}h ({lead_h//24}d)")
        ax.grid(alpha=0.3)
        ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
        ax.tick_params(axis="x", labelsize=8)
        if lead_idx == leads[-1]:
            ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("v22 vs v23 K-scan — aggregate MSE-improvement (paper formula, diffs_stddev normalized)", fontsize=12)
    plt.tight_layout()
    out_path = OUT_DIR / "kscan_summary_v22_vs_v23.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")
    print()
    print("=== aggregate MSE-imp @ 240h table ===")
    print(f"{'K':>4}  {'v22':>10}  {'v23-20k':>10}  {'v23-23k':>10}")
    for K in [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]:
        v22_v = mse_imp(v22[K], 39) if K in v22 else None
        v20_v = mse_imp(v23_20k[K], 39) if K in v23_20k else None
        v23_v = mse_imp(v23_23k[K], 39) if K in v23_23k else None
        print(f"{K:>4}  "
              f"{(f'{v22_v:>10.2f}%' if v22_v is not None else '       n/a')}  "
              f"{(f'{v20_v:>10.2f}%' if v20_v is not None else '       n/a')}  "
              f"{(f'{v23_v:>10.2f}%' if v23_v is not None else '       n/a')}")


if __name__ == "__main__":
    main()
