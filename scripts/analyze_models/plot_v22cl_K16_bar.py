"""K=16 bar chart: open cold_bp vs cl cold_bp vs cl cold_full vs cl warm_full
at L40 (240h), per variable. Highlights the deployment-mode winner."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CL_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-27-v22cl-kscan-eval")
CL_WARM_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-28-v22cl-kscan-warm")
OPEN_BASE = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-25-Res2Mamba")
OPEN_FILL = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-27-v22-open-kfill-eval")
OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-06-28-v22cl-kscan/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARS = [
    ("2m_temperature",          "per_variable_per_step", "2m_T"),
    ("10m_u_component_of_wind", "per_variable_per_step", "10m_u"),
    ("geopotential_level500",   "per_channel_per_step",  "z500"),
    ("temperature_level850",    "per_channel_per_step",  "t850"),
]

def imp_at(p, var, src, k=39):
    if p is None or not Path(p).exists(): return None
    d = json.loads(Path(p).read_text())
    s = d.get(src, {}).get(var)
    if s is None: return None
    arr = s.get("improvement_pct_rmse") or s.get("improvement_pct")
    return arr[k] if arr and len(arr) > k else None


def open_loop_path(K):
    for p in [OPEN_BASE / f"K{K}_step20000_cold_bp.json",
              OPEN_BASE / f"K{K}_step6000_cold_bp.json",
              OPEN_BASE / f"K{K}_step5000_cold_bp.json",
              OPEN_BASE / f"K{K}_step3000_cold_bp.json",
              OPEN_BASE / f"K{K}_step2000_cold_bp.json",
              OPEN_FILL / f"v22_open_K{K}_step2000_cold_bp_zero.json"]:
        if p.exists(): return p
    return None


def gather(K):
    rows = {}
    rows["open cold_bp"]  = open_loop_path(K)
    rows["cl cold_bp"]    = CL_DIR  / f"v22cl_K{K}_step2000_cold_bp_zero.json"
    rows["cl cold_full"]  = CL_DIR  / f"v22cl_K{K}_step2000_cold_full_zero.json"
    rows["cl warm_bp"]    = CL_WARM_DIR / f"v22cl_K{K}_step2000_warm_bp_W24_zero.json"
    rows["cl warm_full"]  = CL_WARM_DIR / f"v22cl_K{K}_step2000_warm_full_W24_zero.json"
    return rows


def plot_one_K(K: int, out_path: Path):
    rows = gather(K)
    modes = list(rows.keys())
    colors = {"open cold_bp": "#777", "cl cold_bp": "#3b82f6",
              "cl cold_full": "#16a34a", "cl warm_bp": "#f59e0b",
              "cl warm_full": "#ef4444"}

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(VARS))
    bw = 0.16
    offsets = np.arange(len(modes)) - (len(modes) - 1) / 2.0

    for i, mode in enumerate(modes):
        vals = []
        for var, src, _ in VARS:
            v = imp_at(rows[mode], var, src, 39)
            vals.append(v if v is not None else 0.0)
        bars = ax.bar(x + offsets[i] * bw, vals, bw,
                      color=colors[mode], label=mode,
                      edgecolor="black", linewidth=0.5)
        # Annotate values on top of bars
        for j, v in enumerate(vals):
            if v != 0:
                ax.text(x[j] + offsets[i] * bw, v + (0.3 if v > 0 else -0.6),
                        f"{v:+.2f}", ha="center", va="bottom" if v > 0 else "top",
                        fontsize=8, fontweight="bold")

    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([v[2] for v in VARS])
    ax.set_ylabel("improvement vs GC baseline (%)  @ L40 (240h)")
    ax.set_title(f"v22cl K={K} step2000 res=2 — deployment mode comparison at lead 240h",
                 fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    # K=16 highlight (user-requested), plus K=14 and K=18 for context
    for K in [14, 16, 18]:
        plot_one_K(K, OUT_DIR / f"v22cl_K{K}_modes_bar_L40.png")
