"""Per-variable K-scan comparison for res=1 v22 closed-loop vs published v22 open-loop.

Compares:
  - open cold_bp: existing v22 res=1 eval JSONs in 2026-05-23-v22/eval_jsons/
                  (these used the original ckpt-init; for fair comparison we
                  could also use the 2026-06-26-v22-kscan-zero-init/ versions)
  - cl cold_bp / cl cold_full: v22cl_res1 eval JSONs in 2026-6-28-v22cl-r1-eval/

Outputs:
  v22cl_res1_kscan_per_var_L40.png       — 4 vars × imp% vs K at L40
  v22cl_res1_kscan_per_var_vs_lead.png   — 4 vars × lead-curve for selected K
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-06-28-v22cl-r1-kscan/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CL_DIR     = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-28-v22cl-r1-eval")
OPEN_BASE  = Path("/home/lm8598/Weather_Global_experiments/results/2026-05-23-v22/eval_jsons")
OPEN_ZERO  = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-26-v22-kscan-zero-init")

KS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
SELECTED_K = [1, 8, 14, 18, 22]

VARS = [
    ("2m_temperature",          "per_variable_per_step", "2m_T"),
    ("10m_u_component_of_wind", "per_variable_per_step", "10m_u"),
    ("geopotential_level500",   "per_channel_per_step",  "z500"),
    ("temperature_level850",    "per_channel_per_step",  "t850"),
]


def load_eval(p):
    if p is None or not Path(p).exists(): return None
    return json.loads(Path(p).read_text())


def open_loop_path(K):
    # Prefer the zero-init re-eval to match closed-loop's zero-init eval.
    candidates = [
        OPEN_ZERO / f"v22_K{K}_K40_zero_init.json",
        OPEN_BASE / f"v22_K{K}_K40.json",
    ]
    for p in candidates:
        if p.exists(): return p
    return None


def closed_loop_path(K, mode):
    return CL_DIR / f"v22cl_r1_K{K}_{mode}_zero.json"


def imp_at(ev, var, src, k):
    if ev is None: return None
    s = ev.get(src, {})
    if var not in s: return None
    arr = s[var].get("improvement_pct_rmse") or s[var].get("improvement_pct")
    return arr[k] if arr and len(arr) > k else None


def plot_per_var_vs_K(out_path: Path, lead_idx: int = 39, lead_label: str = "L40 (240h)"):
    modes_to_plot = [
        ("open_cold_bp", "open cold_bp",   "o-",  "k"),
        ("cl_cold_bp",   "cl cold_bp",     "s--", "C0"),
        ("cl_cold_full", "cl cold_full",   "D-",  "C2"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for i, (var_name, src_key, var_label) in enumerate(VARS):
        ax = axes[i]
        for mode, label, style, color in modes_to_plot:
            xs, ys = [], []
            for K in KS:
                if mode == "open_cold_bp":
                    p = open_loop_path(K)
                else:
                    m = mode.replace("cl_", "")
                    p = closed_loop_path(K, m)
                ev = load_eval(p)
                v = imp_at(ev, var_name, src_key, lead_idx)
                if v is None: continue
                xs.append(K); ys.append(v)
            if xs:
                ax.plot(xs, ys, style, color=color, ms=5, lw=1.8, label=label, alpha=0.9)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(KS); ax.set_xlabel("K (v22 paper label)")
        ax.set_ylabel("improvement vs GC baseline (%)")
        ax.set_title(f"{var_label} @ {lead_label}", fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="best")
    fig.suptitle("v22 K-scan res=1: open-loop vs closed-loop sg @ step 26000 (per variable)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved {out_path}")
    plt.close(fig)


def print_summary_table():
    print()
    print("=" * 100)
    print("res=1 K-scan: imp% at L40 by K")
    print("=" * 100)
    for var_name, src_key, var_label in VARS:
        print(f"\n## {var_label}")
        print(f"  {'K':>3s}  {'open cold_bp':>13s}  {'cl cold_bp':>12s}  {'cl cold_full':>14s}")
        for K in KS:
            cells = []
            for mode in ["open_cold_bp", "cl_cold_bp", "cl_cold_full"]:
                if mode == "open_cold_bp":
                    p = open_loop_path(K)
                else:
                    m = mode.replace("cl_", "")
                    p = closed_loop_path(K, m)
                ev = load_eval(p)
                v = imp_at(ev, var_name, src_key, 39)
                cells.append(v)
            line = f"  K={K:<2d}"
            for v, w in zip(cells, [13, 12, 14]):
                if v is None:
                    line += f"  {'n/a':>{w}s}"
                else:
                    line += f"  {v:>+{w-1}.2f}%"
            print(line)


if __name__ == "__main__":
    print_summary_table()
    plot_per_var_vs_K(OUT_DIR / "v22cl_res1_kscan_per_var_L40.png",
                      lead_idx=39, lead_label="L40 (240h)")
