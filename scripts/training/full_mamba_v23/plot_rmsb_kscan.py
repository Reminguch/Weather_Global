"""Plot v22 K-sweep RMSB improvement: cold_bp (open-loop) vs warm_full (closed-loop).

Reads JSON output of patched eval_v22_clean.py which now writes
  per_variable_per_step[var]['rmsb_baseline'] (K-length list)
  per_variable_per_step[var]['rmsb_full']     (K-length list)
  per_variable_per_step[var]['improvement_pct_rmsb']  (K-length list)
and per_channel_per_step[var_levelN] same fields for atmospheric vars.

Output layout matches results/2026-05-23-v22/plots/v22_vs_baseline.png:
  Two-panel figure per mode (= per eval feedback):
    Left:  improvement % vs lead (h), one curve per training K
    Right: improvement % vs K, one curve per (selected lead)

We aggregate by averaging the 11 main channels (or specifically over surface vars).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --------------- aggregation helpers ---------------

SURFACE_VARS = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation_6hr",
]


def aggregate_rmsb_improvement(d: dict, agg: str = "surface_mean") -> np.ndarray:
    """Returns shape (n_lead,) improvement % aggregated across selected channels."""
    pv = d.get("per_variable_per_step", {})
    if agg == "2m_temperature":
        e = pv.get("2m_temperature", {})
        return np.asarray(e.get("improvement_pct_rmsb", []), dtype=float)
    if agg == "surface_mean":
        arrs = []
        for v in SURFACE_VARS:
            e = pv.get(v, {})
            arr = e.get("improvement_pct_rmsb")
            if arr is None: continue
            arrs.append(np.asarray(arr, dtype=float))
        if not arrs:
            return np.zeros(0)
        return np.mean(np.stack(arrs, axis=0), axis=0)
    raise ValueError(f"unknown agg={agg}")


# --------------- plotting ---------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir",
                   default="/home/lm8598/Weather_Global_experiments/results/2026-06-15-v22-rmsb-kscan")
    p.add_argument("--out-png", required=True)
    p.add_argument("--mode", choices=["cold_bp", "warm_full"], required=True)
    p.add_argument("--agg", choices=["surface_mean", "2m_temperature"], default="surface_mean")
    p.add_argument("--ks", default="1,2,4,6,8,10,12,14,16,18,20,22",
                   help="Comma-separated K values to plot.")
    p.add_argument("--leads-h", default="6,24,48,72,120,168,240",
                   help="Lead times (hours) for right-panel K-scan.")
    return p.parse_args()


def main():
    cfg = parse_args()
    in_dir = Path(cfg.in_dir)
    Ks = [int(k) for k in cfg.ks.split(",")]
    leads_h_target = [int(h) for h in cfg.leads_h.split(",")]

    # Gather: improvement[K] = (n_lead,) array
    series: dict[int, np.ndarray] = {}
    for K in Ks:
        fname = in_dir / f"v22_K{K}_{cfg.mode}.json"
        if not fname.exists():
            print(f"MISSING: {fname}")
            continue
        try:
            d = json.load(open(fname))
            arr = aggregate_rmsb_improvement(d, agg=cfg.agg)
            if len(arr) == 0:
                print(f"  {fname.name}: no rmsb data yet (need patched eval)")
                continue
            series[K] = arr
        except Exception as e:
            print(f"  {fname.name}: error {e}")

    if not series:
        print("No data to plot. Exit.")
        return

    # All series should have same n_lead = 40 → leads at 6h..240h
    n_lead = max(len(arr) for arr in series.values())
    leads_h = (np.arange(n_lead) + 1) * 6   # 6, 12, ..., 240

    # --- Plot ---
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: improvement vs lead time, one curve per K
    cmap = plt.cm.viridis
    K_min, K_max = min(series), max(series)
    for K in sorted(series):
        arr = series[K]
        color = cmap((K - K_min) / max(K_max - K_min, 1))
        ax_l.plot(leads_h[:len(arr)], arr, "-o", color=color, lw=1.2, markersize=3,
                  label=f"K={K}")
    ax_l.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax_l.set_xlabel("lead time (h)")
    ax_l.set_ylabel("RMSB improvement % vs baseline")
    ax_l.set_title(f"v22 ({cfg.mode}, agg={cfg.agg}): RMSB improvement vs lead, by K")
    ax_l.grid(alpha=0.3)
    ax_l.legend(loc="best", fontsize=7, ncol=2)

    # Right: K-scan at multiple leads
    # For each selected lead h: collect (K, improvement_at_h)
    cmap2 = plt.cm.plasma
    n_leads_sel = len(leads_h_target)
    for li, h in enumerate(leads_h_target):
        k_idx = h // 6 - 1   # 0-indexed lead step
        xs = sorted(series)
        ys = [series[K][k_idx] if k_idx < len(series[K]) else np.nan for K in xs]
        color = cmap2(li / max(n_leads_sel - 1, 1))
        ax_r.plot(xs, ys, "-o", color=color, lw=1.4, markersize=5, label=f"lead {h}h")
    ax_r.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax_r.set_xlabel("K (training AR-tail length)")
    ax_r.set_ylabel("RMSB improvement % vs baseline")
    ax_r.set_title(f"v22 ({cfg.mode}): K-scan, multiple leads")
    ax_r.grid(alpha=0.3)
    ax_r.legend(loc="best", fontsize=8)

    plt.tight_layout()
    Path(cfg.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg.out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {cfg.out_png}")


if __name__ == "__main__":
    main()
