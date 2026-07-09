#!/usr/bin/env python
"""v22 warm_bp K-scan — same 2-panel layout as plot_kscan_overlay.py / the
existing v22 K-scan plot, but using warm_bp eval JSONs (clean state init +
24-step truth warmup, bp-feedback).

LEFT:  paper-formula weighted MSE improvement % vs lead time, 12 K curves.
RIGHT: improvement % vs K for selected leads.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm

EVAL_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/"
                "results/2026-06-01-v22clean/eval_jsons")
OUT_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/"
               "results/2026-06-01-v22clean/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
SELECTED_LEADS_H = [6, 24, 48, 72, 120, 168, 240]

# Paper-formula variable weights (same as plot_v22_kscan_vs_K1.py)
W_VAR = {
    "10m_u_component_of_wind": 0.1, "10m_v_component_of_wind": 0.1,
    "2m_temperature": 1.0, "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1, "geopotential": 1.0,
    "temperature": 1.0, "u_component_of_wind": 1.0,
    "v_component_of_wind": 1.0, "vertical_velocity": 1.0,
    "specific_humidity": 1.0,
}
sigma_ds = xr.open_dataset(
    "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/"
    "stats/diffs_stddev_by_level.nc")
SIGMA = {v: float(sigma_ds[v].values if sigma_ds[v].values.ndim == 0
                  else sigma_ds[v].values.mean())
         for v in W_VAR if v in sigma_ds}


def weighted_mse(ev: dict, k_idx: int, key: str) -> float:
    """key='rmse_baseline' or 'rmse_full'. Paper-formula weighted MSE."""
    total = 0.0
    for v in ev["per_variable_per_step"]:
        if v not in SIGMA:
            continue
        w = W_VAR.get(v, 1.0)
        s = SIGMA[v]
        r = ev["per_variable_per_step"][v][key][k_idx]
        total += w * r ** 2 / s ** 2
    return total


def load_K(K: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (weighted_mse_baseline[lead], weighted_mse_full[lead]) for this K."""
    p = EVAL_DIR / f"v22_K{K}_warm_bp.json"
    if not p.exists():
        print(f"MISSING {p}")
        return None, None
    ev = json.load(open(p))
    n_lead = ev["target_steps"]
    base = np.array([weighted_mse(ev, k, "rmse_baseline") for k in range(n_lead)])
    full = np.array([weighted_mse(ev, k, "rmse_full") for k in range(n_lead)])
    return base, full


def main() -> None:
    per_K = {}
    for K in KS:
        base, full = load_K(K)
        if base is None:
            continue
        per_K[K] = (base, full)
    if not per_K:
        print("no data")
        return

    # All K's baseline should be ≈ same (same anchors, clean init). Use K=22's baseline as canonical.
    leads_h = np.arange(1, len(per_K[KS[-1]][0]) + 1) * 6

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6))
    cmap = cm.get_cmap("viridis")
    norm = plt.Normalize(min(per_K), max(per_K))

    # ---- LEFT: improvement vs lead time, all K ----
    for K in sorted(per_K):
        base, full = per_K[K]
        imp = (1 - full / base) * 100
        ax_l.plot(leads_h, imp, "-o", ms=4, lw=1.5, color=cmap(norm(K)),
                  label=f"v22 K={K}")
    ax_l.axhline(0, color="k", lw=0.5)
    ax_l.set_xlabel("lead time (h)")
    ax_l.set_ylabel("improvement vs baseline (%)")
    ax_l.set_title("v22 warm_bp (chunk=24, d_conv=4) vs GraphCast baseline — lead 6h..240h")
    ax_l.grid(alpha=0.3)
    ax_l.legend(loc="best", fontsize=7, ncol=2)

    # ---- RIGHT: improvement vs K for selected leads ----
    cmap2 = cm.get_cmap("tab10")
    Ks_sorted = sorted(per_K)
    for i, lh in enumerate(SELECTED_LEADS_H):
        k_idx = lh // 6 - 1
        imps = [(1 - per_K[K][1][k_idx] / per_K[K][0][k_idx]) * 100 for K in Ks_sorted]
        ax_r.plot(Ks_sorted, imps, "-o", ms=5, lw=1.8, color=cmap2(i),
                  label=f"lead {lh}h")
    ax_r.axhline(0, color="k", lw=0.5)
    ax_r.set_xlabel("K (training AR tail)")
    ax_r.set_ylabel("improvement (%)")
    ax_r.set_title("v22 warm_bp K-scan")
    ax_r.set_xticks(Ks_sorted)
    ax_r.grid(alpha=0.3)
    ax_r.legend(loc="best", fontsize=9)

    fig.suptitle(
        "v22 warm_bp (24-step truth warmup, bp-feedback, clean state init) vs GraphCast baseline "
        "— paper formula, lead 6h..240h (10 days)", fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / "v22_warm_bp_kscan_2panel.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)

    # Print improvement table
    print("\n=== v22 warm_bp improvement vs baseline (%) ===")
    print(f"{'K':>4}", *[f"{lh:>5}h" for lh in SELECTED_LEADS_H])
    for K in Ks_sorted:
        base, full = per_K[K]
        imps = [(1 - full[lh//6 - 1] / base[lh//6 - 1]) * 100 for lh in SELECTED_LEADS_H]
        print(f"{K:>4}", *[f"{imp:>+5.2f}" for imp in imps])


if __name__ == "__main__":
    main()
