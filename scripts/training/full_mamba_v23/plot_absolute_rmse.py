#!/usr/bin/env python
"""Absolute-RMSE plots in physical units (K, m/s, m^2/s^2) — answers Ilya's
question: "GC baseline at long lead is already bad; what does the %
improvement mean in absolute terms?"

Per variable, per pressure level (for 3D vars), plot:
  Top:    raw lat-weighted RMSE vs lead, lines = baseline + each model
  Bottom: absolute reduction (baseline_rmse - model_rmse) vs lead
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

OUT_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/"
               "results/2026-05-30-v23/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

V22_DIR = "/home/lm8598/Weather_Global_experiments/results/2026-05-23-v22/eval_jsons"
V23F_DIR = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0526_v23_eval_K40"

# Variables in eval JSON (top-level keys of per_variable_per_step)
SURFACE_VARS = [
    ("2m_temperature", "K"),
    ("mean_sea_level_pressure", "Pa"),
    ("10m_u_component_of_wind", "m/s"),
    ("10m_v_component_of_wind", "m/s"),
    ("total_precipitation_6hr", "m"),
]
# 3D vars: rmse arrays have shape (lead, level), reduce by mean over levels
THREED_VARS = [
    ("temperature", "K"),
    ("geopotential", "m^2/s^2"),
    ("u_component_of_wind", "m/s"),
    ("v_component_of_wind", "m/s"),
    ("specific_humidity", "kg/kg"),
]

KS_v22 = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
KS_v23 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]


def get_rmse(ev: dict, var: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (rmse_baseline, rmse_full) of shape (n_lead,) for this variable.
    For 3D vars, mean over pressure levels."""
    d = ev["per_variable_per_step"][var]
    rb = np.asarray(d["rmse_baseline"])
    rf = np.asarray(d["rmse_full"])
    if rb.ndim == 2:  # (lead, level)
        rb = rb.mean(axis=-1)
        rf = rf.mean(axis=-1)
    return rb, rf


def load_run(path: str, var: str):
    if not Path(path).exists():
        return None
    return get_rmse(json.load(open(path)), var)


def plot_var(var: str, unit: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9),
                              gridspec_kw={"height_ratios": [1.4, 1]},
                              sharex=True)
    # Left col = v22 K-scan, Right col = v23-fresh K-scan
    for col, (label, ks, dir_, pattern) in enumerate([
        ("v22 (d_conv=4)", KS_v22, V22_DIR, "v22_K{K}_K40.json"),
        ("v23 fresh (d_conv=8)", KS_v23, V23F_DIR, "v23_K{K}_fresh_K40.json"),
    ]):
        ax_top, ax_bot = axes[0, col], axes[1, col]
        cmap = cm.get_cmap("viridis")
        norm = plt.Normalize(min(ks), max(ks))
        baseline_plotted = False

        for K in ks:
            d = load_run(f"{dir_}/{pattern.format(K=K)}", var)
            if d is None:
                continue
            rb, rf = d
            leads_h = np.arange(1, len(rb) + 1) * 6
            if not baseline_plotted:
                ax_top.plot(leads_h, rb, "k-", lw=2.0, label="GraphCast baseline")
                baseline_plotted = True
            ax_top.plot(leads_h, rf, color=cmap(norm(K)), lw=1.4, alpha=0.85,
                        label=f"K={K}")
            # Absolute reduction = baseline - model. Positive = model is better.
            ax_bot.plot(leads_h, rb - rf, color=cmap(norm(K)), lw=1.4, alpha=0.85)

        ax_top.set_title(f"{label}")
        ax_top.set_ylabel(f"lat-weighted RMSE ({unit})")
        ax_top.grid(alpha=0.3)
        ax_top.legend(fontsize=7, loc="upper left", ncol=2)

        ax_bot.axhline(0, color="k", lw=0.7)
        ax_bot.set_ylabel(f"baseline − model RMSE ({unit})")
        ax_bot.set_xlabel("lead time (h)")
        ax_bot.grid(alpha=0.3)

    fig.suptitle(
        f"{var} — absolute RMSE in physical units, K-scan vs lead. "
        f"Top row: raw RMSE; bottom row: absolute reduction vs GC baseline.",
        fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / f"abs_rmse_{var}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


def plot_all() -> None:
    for var, unit in SURFACE_VARS + THREED_VARS:
        try:
            plot_var(var, unit)
        except KeyError as e:
            print(f"skip {var}: {e}")


if __name__ == "__main__":
    plot_all()
