"""K-scan comparison: open-loop vs closed-loop sg at res=2.

For each K in [1,2,4,6,8,10,12,14,16,18,20,22] at step 2000, plot:
  - LEFT  panel: improvement % vs lead (paper-weighted MSE), all K curves
                 — open-loop cold_bp + closed-loop cold_bp + closed-loop cold_full
  - RIGHT panel: improvement % vs K at selected leads (24h, 72h, 120h, 168h, 240h)

Output: 2026-06-28-v22cl-kscan/plots/
  - open_vs_closed_cold_bp_kscan.png
  - closed_loop_cold_bp_vs_cold_full_kscan.png
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import xarray as xr

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

KS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-06-28-v22cl-kscan/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Source eval JSONs
CL_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-27-v22cl-kscan-eval")
OPEN_BASE = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-25-Res2Mamba")
OPEN_FILL = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-27-v22-open-kfill-eval")


def mse_imp(ev, k_idx):
    tb = tf = 0.0
    for v in ev.get("per_variable_per_step", {}):
        if v not in SIGMA: continue
        w = W_VAR.get(v, 1.0); s = SIGMA[v]
        rb = ev["per_variable_per_step"][v]["rmse_baseline"][k_idx]
        rf = ev["per_variable_per_step"][v]["rmse_full"][k_idx]
        tb += w * rb**2 / s**2; tf += w * rf**2 / s**2
    return (1 - tf / tb) * 100 if tb > 0 else 0


def load_eval(p):
    if not Path(p).exists(): return None
    return json.loads(Path(p).read_text())


def open_loop_path(K):
    # K∈{2,4,8,10,14,18}: in 2026-6-25-Res2Mamba/K{K}_step20000_cold_bp.json
    # K=1: in 2026-6-25-Res2Mamba/K1_*  ? probably K1_step3000_cold_bp.json
    # K∈{6,12,16,20,22}: in 2026-6-27-v22-open-kfill-eval/v22_open_K{K}_step2000_cold_bp_zero.json
    # Try several plausible names.
    candidates = [
        OPEN_BASE / f"K{K}_step20000_cold_bp.json",
        OPEN_BASE / f"K{K}_step6000_cold_bp.json",
        OPEN_BASE / f"K{K}_step5000_cold_bp.json",
        OPEN_BASE / f"K{K}_step4000_cold_bp.json",
        OPEN_BASE / f"K{K}_step3000_cold_bp.json",
        OPEN_BASE / f"K{K}_step2000_cold_bp.json",
        OPEN_BASE / f"K{K}_step1000_cold_bp.json",
        OPEN_FILL / f"v22_open_K{K}_step2000_cold_bp_zero.json",
    ]
    for p in candidates:
        if p.exists(): return p
    return None


def closed_loop_path(K, mode):
    return CL_DIR / f"v22cl_K{K}_step2000_{mode}_zero.json"


def plot_kscan_overlay(out_path: Path, include_modes=("open_cold_bp", "cl_cold_bp", "cl_cold_full")):
    """Two-panel K-scan overlay across modes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # Pull evs
    series = {  # mode_label → list of (K, ev)
        "open_cold_bp":  [],
        "cl_cold_bp":    [],
        "cl_cold_full":  [],
    }
    for K in KS:
        for mode in include_modes:
            if mode == "open_cold_bp":
                p = open_loop_path(K)
            else:
                cl_mode = "cold_bp" if mode == "cl_cold_bp" else "cold_full"
                p = closed_loop_path(K, cl_mode)
            ev = load_eval(p) if p else None
            if ev: series[mode].append((K, ev))

    # n_lead
    any_ev = next((ev for vals in series.values() for _K, ev in vals), None)
    if any_ev is None:
        print("No evals found"); return
    n_lead = any_ev["target_steps"]
    lead_h = [6 * (i + 1) for i in range(n_lead)]

    SELECTED = [24, 72, 120, 168, 240]
    available = [h for h in SELECTED if h <= max(lead_h)]

    MODE_STYLE = {
        "open_cold_bp": dict(linestyle="-",  marker="o", cmap="Greys",  label_prefix="open cold_bp K=",   alpha=1.0),
        "cl_cold_bp":   dict(linestyle="-",  marker="s", cmap="Blues",  label_prefix="cl cold_bp K=",     alpha=1.0),
        "cl_cold_full": dict(linestyle="--", marker="D", cmap="Greens", label_prefix="cl cold_full K=",  alpha=1.0),
    }

    # LEFT panel: imp% vs lead
    ax = axes[0]
    for mode in include_modes:
        cmap = cm.get_cmap(MODE_STYLE[mode]["cmap"])
        Ks_here = [K for K, _ in series[mode]]
        if not Ks_here: continue
        norm = plt.Normalize(min(Ks_here) - 2, max(Ks_here) + 2)
        for K, ev in series[mode]:
            imps = [mse_imp(ev, k) for k in range(n_lead)]
            ax.plot(lead_h, imps, MODE_STYLE[mode]["linestyle"],
                    marker=MODE_STYLE[mode]["marker"], ms=3, lw=1.4,
                    color=cmap(norm(K)),
                    label=MODE_STYLE[mode]["label_prefix"] + f"{K}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lead time (h)"); ax.set_ylabel("improvement vs baseline (%)")
    ax.set_title(f"K-scan res=2: open-loop vs closed-loop sg")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.42))

    # RIGHT panel: imp% vs K at selected leads
    ax = axes[1]
    for mode in include_modes:
        for i, lh in enumerate(available):
            k_idx = lh // 6 - 1
            Ks_here = [K for K, _ in series[mode]]
            imps = [mse_imp(ev, k_idx) for _K, ev in series[mode]]
            ax.plot(Ks_here, imps,
                    MODE_STYLE[mode]["linestyle"],
                    marker=MODE_STYLE[mode]["marker"], ms=5, lw=1.7,
                    label=f"{mode.replace('_', ' ')} @ {lh}h")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("K (training AR tail)"); ax.set_ylabel("improvement (%)")
    ax.set_title("K-scan @ selected leads")
    ax.set_xticks(KS); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.42))

    fig.suptitle("v22 K-scan res=2: open-loop vs closed-loop sg (step2000) — paper-weighted MSE imp%",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")
    plt.close(fig)


def print_summary_table():
    print()
    print("=" * 100)
    print("K-scan summary: improvement_pct at L40 by K (paper-weighted MSE)")
    print("=" * 100)
    print(f"  {'K':>3s}  {'open cold_bp':>14s}  {'cl cold_bp':>12s}  {'cl cold_full':>14s}")
    for K in KS:
        po = open_loop_path(K); pcb = closed_loop_path(K, "cold_bp"); pcf = closed_loop_path(K, "cold_full")
        eo = load_eval(po) if po else None
        ecb = load_eval(pcb); ecf = load_eval(pcf)
        line = f"  K={K:<2d}"
        for ev in (eo, ecb, ecf):
            if ev is None:
                line += "  " + " " * 12 + "n/a"
            else:
                imp = mse_imp(ev, 39)  # L40
                line += f"  {imp:>+11.2f}%"
        print(line)


if __name__ == "__main__":
    print_summary_table()
    plot_kscan_overlay(OUT_DIR / "v22cl_vs_open_kscan.png",
                       include_modes=("open_cold_bp", "cl_cold_bp", "cl_cold_full"))
