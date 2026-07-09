#!/usr/bin/env python
"""v22 K-scan plot using v22 K=1 as baseline (instead of frozen DeepMind GC).

This makes v22 K-scan directly comparable to GCv1 K-scan, which already
uses "vs K=1" as baseline. The question both plots now answer is:
  "how much does K-curriculum training help on top of K=1?"

Outputs 2 plots:
  1. v22 K-scan alone (v22-style 2-panel, but baseline = v22 K=1)
  2. v22 vs GCv1 side-by-side comparison (4-panel)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import cm

V22_DIR = "/home/lm8598/Weather_Global_experiments/results/2026-05-23-v22/eval_jsons"
GCV1_EVAL_ROOT = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/GCv1_eval")
OUT_DIR = Path(
    "/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/"
    "results/2026-06-01-GCv1/plots"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paper variable weights (same as plot_kscan_overlay.py / plot_kscan_summary.py)
W_VAR = {
    "10m_u_component_of_wind": 0.1, "10m_v_component_of_wind": 0.1,
    "2m_temperature": 1.0, "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1, "geopotential": 1.0,
    "temperature": 1.0, "u_component_of_wind": 1.0,
    "v_component_of_wind": 1.0, "vertical_velocity": 1.0, "specific_humidity": 1.0,
}
sigma_ds = xr.open_dataset(
    "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/"
    "stats/diffs_stddev_by_level.nc")
SIGMA = {v: float(sigma_ds[v].values if sigma_ds[v].values.ndim == 0
                  else sigma_ds[v].values.mean())
         for v in W_VAR if v in sigma_ds}


def v22_weighted_mse(ev: dict, k_idx: int) -> float:
    """Paper-formula weighted MSE for v22 using rmse_full (the corrected model)."""
    total = 0.0
    for v in ev["per_variable_per_step"]:
        if v not in SIGMA:
            continue
        w = W_VAR.get(v, 1.0)
        s = SIGMA[v]
        rf = ev["per_variable_per_step"][v]["rmse_full"][k_idx]
        total += w * rf ** 2 / s ** 2
    return total


def load_v22_kscan() -> dict[int, np.ndarray]:
    """Return {K: weighted_mse[lead_idx]} for v22, using rmse_full."""
    KS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    out = {}
    for K in KS:
        p = Path(V22_DIR) / f"v22_K{K}_K40.json"
        if not p.exists():
            print(f"missing {p}")
            continue
        ev = json.load(open(p))
        n_lead = ev["target_steps"]
        out[K] = np.array([v22_weighted_mse(ev, k_idx) for k_idx in range(n_lead)])
    return out


def load_gcv1_kscan() -> dict[int, np.ndarray]:
    """Return {K: weighted_allvars[lead_idx]} for GCv1 using weighted_allvars metric."""
    KS = list(range(1, 19))
    out = {}
    for K in KS:
        csv = GCV1_EVAL_ROOT / f"K{K}" / f"K{K}_eval.csv"
        if not csv.exists():
            print(f"missing {csv}")
            continue
        df = pd.read_csv(csv)
        w = df[(df["metric_kind"] == "weighted_allvars") & (df["eval_mode"] == "cold")]
        w = w.sort_values("lead_steps").reset_index(drop=True)
        out[K] = w["value"].values
    return out


SELECTED_LEADS_H = [6, 24, 48, 72, 120, 168, 240]


def plot_kscan_2panel(per_K: dict[int, np.ndarray], label: str, out_path: Path,
                      title_extra: str = "") -> None:
    KS = sorted(per_K.keys())
    mse_K1 = per_K[1]
    leads_h = np.arange(1, len(mse_K1) + 1) * 6

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6))
    cmap = cm.get_cmap("viridis")
    norm = plt.Normalize(min(KS), max(KS))

    for K in KS:
        imp = (1 - per_K[K] / mse_K1) * 100
        ax_l.plot(leads_h, imp, "-o", ms=4, lw=1.5, color=cmap(norm(K)),
                  label=f"{label} K={K}")
    ax_l.axhline(0, color="k", lw=0.5)
    ax_l.set_xlabel("lead time (h)")
    ax_l.set_ylabel("improvement vs K=1 baseline (%)")
    ax_l.set_title(f"{label} — improvement vs K=1{title_extra}")
    ax_l.grid(alpha=0.3)
    ax_l.legend(loc="upper left", fontsize=7, ncol=2)

    cmap2 = cm.get_cmap("tab10")
    for i, lh in enumerate(SELECTED_LEADS_H):
        k_idx = lh // 6 - 1
        imps = [(1 - per_K[K][k_idx] / mse_K1[k_idx]) * 100 for K in KS]
        ax_r.plot(KS, imps, "-o", ms=5, lw=1.8, color=cmap2(i),
                  label=f"lead {lh}h")
    ax_r.axhline(0, color="k", lw=0.5)
    ax_r.set_xlabel("K (training AR tail)")
    ax_r.set_ylabel("improvement vs K=1 (%)")
    ax_r.set_title(f"{label} K-scan{title_extra}")
    ax_r.set_xticks(KS)
    ax_r.grid(alpha=0.3)
    ax_r.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        f"{label} K-scan — improvement vs OWN K=1 (paper-formula weighted MSE){title_extra}",
        fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")
    plt.close(fig)


def plot_side_by_side(v22: dict, gcv1: dict, out_path: Path) -> None:
    """4-panel: row 0 = v22; row 1 = GCv1. col 0 = lead curves; col 1 = K-scan."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    cmap = cm.get_cmap("viridis")
    cmap2 = cm.get_cmap("tab10")

    for row, (label, per_K) in enumerate([("v22 (residual+Mamba, res1)", v22),
                                            ("GCv1 (vanilla GC, res2)", gcv1)]):
        KS = sorted(per_K.keys())
        mse_K1 = per_K[1]
        leads_h = np.arange(1, len(mse_K1) + 1) * 6
        norm = plt.Normalize(min(KS), max(KS))

        ax = axes[row, 0]
        for K in KS:
            imp = (1 - per_K[K] / mse_K1) * 100
            ax.plot(leads_h, imp, "-o", ms=3, lw=1.3, color=cmap(norm(K)),
                    label=f"K={K}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("lead time (h)")
        ax.set_ylabel("improvement vs own K=1 (%)")
        ax.set_title(f"{label}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=6, ncol=3)

        ax = axes[row, 1]
        for i, lh in enumerate(SELECTED_LEADS_H):
            k_idx = lh // 6 - 1
            imps = [(1 - per_K[K][k_idx] / mse_K1[k_idx]) * 100 for K in KS]
            ax.plot(KS, imps, "-o", ms=5, lw=1.8, color=cmap2(i),
                    label=f"lead {lh}h")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("K (training AR tail)")
        ax.set_ylabel("improvement vs K=1 (%)")
        ax.set_title(f"{label} K-scan")
        ax.set_xticks(KS)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "v22 (residual+Mamba, parallel-from-K=1) vs GCv1 (vanilla GC, paper-style cascade) — "
        "BOTH improvements vs OWN K=1 baseline", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")
    plt.close(fig)


def main() -> None:
    v22 = load_v22_kscan()
    gcv1 = load_gcv1_kscan()

    plot_kscan_2panel(v22, "v22 (residual+Mamba)",
                       OUT_DIR / "kscan_v22_vs_v22K1.png",
                       title_extra=" — d_conv=4, all-from-K=1 cascade")
    plot_kscan_2panel(gcv1, "GCv1 (vanilla GC)",
                       OUT_DIR / "kscan_GCv1_vs_GCv1K1.png",
                       title_extra=" — res2-m4-mp6, paper-style sequential cascade")
    plot_side_by_side(v22, gcv1, OUT_DIR / "kscan_v22_vs_GCv1_side_by_side.png")

    # Print comparison table at K=22 / K=18 (max K of each)
    print("\n=== Comparison: max K of each vs own K=1 ===")
    v22_K1, gcv1_K1 = v22[1], gcv1[1]
    print(f"{'lead':>6} {'v22 K=22':>12} {'GCv1 K=18':>12}")
    for lh in SELECTED_LEADS_H:
        k_idx = lh // 6 - 1
        v22_imp = (1 - v22[22][k_idx] / v22_K1[k_idx]) * 100
        gcv1_imp = (1 - gcv1[18][k_idx] / gcv1_K1[k_idx]) * 100
        print(f"{lh:>4}h {v22_imp:>11.2f}% {gcv1_imp:>11.2f}%")


if __name__ == "__main__":
    main()
