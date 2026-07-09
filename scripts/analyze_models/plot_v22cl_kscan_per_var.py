"""Per-variable K-scan comparison: open-loop vs closed-loop sg at res=2.

3 figures:
  1. v22cl_kscan_per_var_L40.png    — 4 vars × imp% vs K at L40 (240h)
  2. v22cl_kscan_per_var_vs_lead.png — 4 vars × imp% vs lead for selected K
  3. v22cl_kscan_warm_modes.png      — adds warm_bp + warm_full curves (when JSONs ready)

Variables: 2m_temperature, 10m_u_component_of_wind, geopotential_level500,
            temperature_level850
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-06-28-v22cl-kscan/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CL_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-27-v22cl-kscan-eval")
CL_WARM_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-28-v22cl-kscan-warm")
OPEN_BASE = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-25-Res2Mamba")
OPEN_FILL = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-27-v22-open-kfill-eval")

KS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
SELECTED_K = [2, 8, 14, 18, 22]

VARS = [
    ("2m_temperature", "per_variable_per_step", "2m_T"),
    ("10m_u_component_of_wind", "per_variable_per_step", "10m_u"),
    ("geopotential_level500", "per_channel_per_step", "z500"),
    ("temperature_level850", "per_channel_per_step", "t850"),
]


def load_eval(p):
    if p is None or not Path(p).exists(): return None
    return json.loads(Path(p).read_text())


def open_loop_path(K):
    candidates = [
        OPEN_BASE / f"K{K}_step20000_cold_bp.json",
        OPEN_BASE / f"K{K}_step6000_cold_bp.json",
        OPEN_BASE / f"K{K}_step5000_cold_bp.json",
        OPEN_BASE / f"K{K}_step4000_cold_bp.json",
        OPEN_BASE / f"K{K}_step3000_cold_bp.json",
        OPEN_BASE / f"K{K}_step2000_cold_bp.json",
        OPEN_FILL / f"v22_open_K{K}_step2000_cold_bp_zero.json",
    ]
    for p in candidates:
        if p.exists(): return p
    return None


def closed_loop_path(K, mode):
    if mode in ("warm_bp", "warm_full"):
        return CL_WARM_DIR / f"v22cl_K{K}_step2000_{mode}_W24_zero.json"
    return CL_DIR / f"v22cl_K{K}_step2000_{mode}_zero.json"


def imp_at(ev, var_name, src_key, lead_idx):
    if ev is None: return None
    src = ev.get(src_key, {})
    if var_name not in src: return None
    arr = src[var_name].get("improvement_pct_rmse") or src[var_name].get("improvement_pct")
    if arr is None or len(arr) <= lead_idx: return None
    return arr[lead_idx]


def collect_series(mode_label):
    """Return {K: imp_array_per_lead} for the given mode_label."""
    out = {}
    for K in KS:
        if mode_label == "open_cold_bp":
            p = open_loop_path(K)
        else:
            mode = mode_label.replace("cl_", "")
            p = closed_loop_path(K, mode)
        ev = load_eval(p)
        if ev is not None:
            out[K] = ev
    return out


def plot_per_var_vs_K(out_path: Path, lead_idx: int = 39, lead_label: str = "L40 (240h)"):
    """4 panels (per var), x=K, y=imp%, lines=modes."""
    modes_to_plot = [
        ("open_cold_bp", "open cold_bp",        "o-", "k"),
        ("cl_cold_bp",   "cl cold_bp",          "s--", "C0"),
        ("cl_cold_full", "cl cold_full",        "D-",  "C2"),
        ("cl_warm_bp",   "cl warm_bp (W=24)",   "v:", "C1"),
        ("cl_warm_full", "cl warm_full (W=24)", "^-",  "C3"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    series_by_mode = {m: collect_series(m) for m, *_ in modes_to_plot}

    for i, (var_name, src_key, var_label) in enumerate(VARS):
        ax = axes[i]
        for mode, label, style, color in modes_to_plot:
            evs = series_by_mode[mode]
            xs, ys = [], []
            for K in KS:
                if K not in evs: continue
                imp = imp_at(evs[K], var_name, src_key, lead_idx)
                if imp is None: continue
                xs.append(K); ys.append(imp)
            if xs:
                ax.plot(xs, ys, style, color=color, ms=5, lw=1.8, label=label, alpha=0.9)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(KS); ax.set_xlabel("K (AR-tail in training)")
        ax.set_ylabel("improvement vs GC baseline (%)")
        ax.set_title(f"{var_label} @ {lead_label}", fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("v22 K-scan res=2: open-loop vs closed-loop sg @ step 2000 (per variable)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved {out_path}")
    plt.close(fig)


def plot_per_var_vs_lead(out_path: Path):
    """4 panels (per var). Each panel: imp% vs lead, lines = (mode, K)."""
    modes_to_plot = [
        ("open_cold_bp", "Greys",  "open cold_bp"),
        ("cl_cold_bp",   "Blues",  "cl cold_bp"),
        ("cl_cold_full", "Greens", "cl cold_full"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    series_by_mode = {m: collect_series(m) for m, *_ in modes_to_plot}

    for i, (var_name, src_key, var_label) in enumerate(VARS):
        ax = axes[i]
        n_lead = None
        for mode, cmap_name, label in modes_to_plot:
            cmap = plt.get_cmap(cmap_name)
            evs = series_by_mode[mode]
            for K in SELECTED_K:
                if K not in evs: continue
                ev = evs[K]
                src = ev.get(src_key, {})
                if var_name not in src: continue
                arr = src[var_name].get("improvement_pct_rmse") or src[var_name].get("improvement_pct")
                if arr is None: continue
                n_lead = len(arr)
                lead_h = [6 * (k + 1) for k in range(n_lead)]
                col_t = (K - min(SELECTED_K)) / max(1, max(SELECTED_K) - min(SELECTED_K))
                ax.plot(lead_h, arr, lw=1.6,
                        color=cmap(0.35 + 0.55 * col_t),
                        label=f"{label} K={K}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("lead time (h)")
        ax.set_ylabel("improvement vs baseline (%)")
        ax.set_title(var_label, fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=3, loc="lower center")
    fig.suptitle("v22 K-scan res=2: imp% vs lead, selected K values (open vs closed-loop sg)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved {out_path}")
    plt.close(fig)


def print_summary_table():
    print()
    print("=" * 110)
    print("Per-variable improvement % at L40 (240h) by K — 4 mode columns")
    print("=" * 110)
    for var_name, src_key, var_label in VARS:
        print(f"\n## {var_label}")
        print(f"  {'K':>3s}  {'open cold_bp':>13s}  {'cl cold_bp':>12s}  {'cl cold_full':>14s}  {'cl warm_bp':>12s}  {'cl warm_full':>14s}")
        for K in KS:
            cells = []
            for mode in ["open_cold_bp", "cl_cold_bp", "cl_cold_full", "cl_warm_bp", "cl_warm_full"]:
                if mode == "open_cold_bp":
                    p = open_loop_path(K)
                else:
                    m = mode.replace("cl_", "")
                    p = closed_loop_path(K, m)
                ev = load_eval(p) if p else None
                v = imp_at(ev, var_name, src_key, 39)
                cells.append(v)
            line = f"  K={K:<2d}"
            for v, w in zip(cells, [13, 12, 14, 12, 14]):
                if v is None:
                    line += f"  {'n/a':>{w}s}"
                else:
                    line += f"  {v:>+{w-1}.2f}%"
            print(line)


if __name__ == "__main__":
    print_summary_table()
    plot_per_var_vs_K(OUT_DIR / "v22cl_kscan_per_var_L40.png", lead_idx=39, lead_label="L40 (240h)")
    plot_per_var_vs_lead(OUT_DIR / "v22cl_kscan_per_var_vs_lead.png")
