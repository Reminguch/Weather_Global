"""K-scan comparison for v22cl res=1 step 14000 vs open-loop baseline.

Standard K-scan view: for each variable, x=K, y=improvement% vs GC baseline
at lead L40 (240h). Three lines per subplot:
  - open cold_bp (v22 published)
  - cl cold_bp   (v22cl closed-loop trained, cold_bp eval)
  - cl cold_full (v22cl closed-loop trained, cold_full eval)

Also a lead-curve view for selected K.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-07-01-v22cl-r1-step14k-eval/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CL_DIR     = Path("/home/lm8598/Weather_Global_experiments/results/2026-07-01-v22cl-r1-step14k-eval")
OPEN_BASE  = Path("/home/lm8598/Weather_Global_experiments/results/2026-05-23-v22/eval_jsons")
OPEN_ZERO  = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-26-v22-kscan-zero-init")

KS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
SELECTED_K = [2, 8, 14, 18, 22]

VARS = [
    ("2m_temperature",          "per_variable_per_step", "2m_T"),
    ("10m_u_component_of_wind", "per_variable_per_step", "10m_u"),
    ("10m_v_component_of_wind", "per_variable_per_step", "10m_v"),
    ("mean_sea_level_pressure", "per_variable_per_step", "mslp"),
]


def load_eval(p):
    if p is None or not Path(p).exists(): return None
    return json.loads(Path(p).read_text())


def open_loop_path(K):
    for p in [
        OPEN_ZERO / f"v22_K{K}_K40_zero_init.json",
        OPEN_BASE / f"v22_K{K}_K40.json",
    ]:
        if p.exists(): return p
    return None


def cl_path(K, mode):
    return CL_DIR / f"v22cl_r1_K{K}_step14000_{mode}_zero.json"


def imp_arr(ev, var, src):
    if ev is None: return None
    s = ev.get(src, {})
    if var not in s: return None
    return s[var].get("improvement_pct_rmse") or s[var].get("improvement_pct")


def imp_at(ev, var, src, k):
    arr = imp_arr(ev, var, src)
    if arr is None or k >= len(arr): return None
    return arr[k]


def plot_per_var_vs_K(lead_idx: int, lead_label: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    modes = [
        ("open_cold_bp", "open cold_bp",  "o-",  "k",  open_loop_path, None),
        ("cl_cold_bp",   "cl cold_bp",    "s-", "C0", cl_path,         "cold_bp"),
        ("cl_cold_full", "cl cold_full",  "D-", "C3", cl_path,         "cold_full"),
    ]
    for i, (var_name, src_key, var_label) in enumerate(VARS):
        ax = axes[i]
        for mode_key, label, style, color, pathfn, mode in modes:
            xs, ys = [], []
            for K in KS:
                p = pathfn(K) if mode is None else pathfn(K, mode)
                ev = load_eval(p)
                v = imp_at(ev, var_name, src_key, lead_idx)
                if v is None: continue
                xs.append(K); ys.append(v)
            if xs:
                ax.plot(xs, ys, style, color=color, ms=6, lw=1.8, label=label, alpha=0.9)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(KS); ax.set_xlabel("K")
        ax.set_ylabel("improvement over GC (%)")
        ax.set_title(f"{var_label} @ {lead_label}", fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="best")
    fig.suptitle("v22cl res=1 step 14000 K-scan (per-variable, L=240h)", fontsize=13)
    plt.tight_layout()
    out = OUT_DIR / f"kscan_L{lead_idx+1}_{lead_label.replace(' ', '_').replace('(','').replace(')','')}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def plot_lead_curves():
    """For selected K, plot improvement% vs lead (24-240h) for 2m_T only."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    modes = [("cold_bp", "cl cold_bp", axes[0]),
             ("cold_full", "cl cold_full", axes[1])]
    cmap = plt.cm.viridis
    for mode, label, ax in modes:
        for i, K in enumerate(SELECTED_K):
            ev = load_eval(cl_path(K, mode))
            arr = imp_arr(ev, "2m_temperature", "per_variable_per_step")
            if arr is None: continue
            leads = np.arange(1, len(arr) + 1) * 6
            ax.plot(leads, arr, "-o", ms=4, color=cmap(i / (len(SELECTED_K) - 1)),
                    label=f"K={K}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("lead (h)")
        if ax is axes[0]:
            ax.set_ylabel("2m_T improvement over GC (%)")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("v22cl res=1 step 14000 — 2m_T improvement vs lead, selected K", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "lead_curves_2m_T.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def print_table():
    print("\n" + "=" * 92)
    print("v22cl res=1 step 14000 — improvement% at 240h by K")
    print("=" * 92)
    for var_name, src_key, var_label in VARS:
        print(f"\n## {var_label}")
        print(f"  {'K':>3s}  {'open':>8s}  {'cl_bp':>8s}  {'cl_full':>8s}")
        for K in KS:
            row = f"  K={K:<2d}"
            for pathfn, mode in [(open_loop_path, None),
                                 (cl_path, "cold_bp"),
                                 (cl_path, "cold_full")]:
                p = pathfn(K) if mode is None else pathfn(K, mode)
                v = imp_at(load_eval(p), var_name, src_key, 39)
                row += f"  {'n/a':>8s}" if v is None else f"  {v:>+7.2f}%"
            print(row)


if __name__ == "__main__":
    print_table()
    plot_per_var_vs_K(lead_idx=39, lead_label="L40 (240h)")
    plot_lead_curves()
