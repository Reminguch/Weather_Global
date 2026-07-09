"""Plot improvement vs lead curves for a warmup-duration sweep on a single K ckpt.

Reads JSONs matching eval_res1_2way_K{K}_step{S}_W{W}.json and plots
improvement % per lead, one curve per W.
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True)
    p.add_argument("--k", type=int, default=12, help="training K for which to compile W-sweep")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main():
    cfg = parse_args()
    OUT = Path(cfg.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    patt = str(Path(cfg.in_dir) / f"eval_res1_2way_K{cfg.k}_step*_W*.json")
    paths = sorted(glob.glob(patt))
    if not paths:
        print(f"no JSON matching {patt}"); return

    data = {}
    for p in paths:
        m = re.search(r"_W(\d+)\.json$", p)
        if not m: continue
        W = int(m.group(1))
        data[W] = json.load(open(p))
        print(f"W={W}: loaded {Path(p).name}")
    Ws = sorted(data.keys())
    cmap = plt.get_cmap("viridis")
    w_color = {W: cmap(i / max(1, len(Ws) - 1)) for i, W in enumerate(Ws)}

    K_t = data[Ws[0]]["target_steps"]
    lead_days = (np.arange(1, K_t + 1) * 6) / 24.0

    surface_vars = ["2m_temperature","total_precipitation_6hr","mean_sea_level_pressure",
                    "10m_u_component_of_wind","10m_v_component_of_wind"]
    upper_vars = ["temperature","specific_humidity","geopotential",
                  "u_component_of_wind","v_component_of_wind"]
    all_vars = surface_vars + upper_vars

    # --- improvement vs lead, multi-W, one panel per var ---
    n = len(all_vars)
    ncols = 5; nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)
    for vi, var in enumerate(all_vars):
        ax = axes[vi // ncols, vi % ncols]
        for W in Ws:
            d = data[W]
            pv = d["per_variable_per_lead"].get(var)
            if pv is None: continue
            ax.plot(lead_days, pv["improvement_gc_mamba_pct"], "-o", lw=1.2, markersize=3,
                    color=w_color[W], label=f"W={W} ({W*6}h)")
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
        ax.set_title(var.replace("_component_of_wind","_wind"), fontsize=10)
        ax.set_xlabel("lead (days)")
        if vi % ncols == 0:
            ax.set_ylabel("improvement % vs baseline (global lat-w RMSE)")
        if vi == 0:
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Warmup-duration sweep on res=1 gc_mamba K={cfg.k} (AR-start aligned across W)",
                 fontsize=12)
    plt.tight_layout()
    out_path = OUT / "W_sweep_improvement_vs_lead.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out_path}")

    # --- lead-window heatmap: rows = W, cols = lead day ---
    days = list(range(1, 11))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for idx, var in enumerate(surface_vars + ["specific_humidity"][:1]):
        ax = axes[idx // 3, idx % 3]
        mat = np.zeros((len(Ws), len(days)))
        for wi, W in enumerate(Ws):
            imp = np.asarray(data[W]['per_variable_per_lead'][var]['improvement_gc_mamba_pct'])
            for di, d in enumerate(days):
                mat[wi, di] = imp[4*(d-1):4*d].mean()
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        ax.set_xticks(range(len(days))); ax.set_xticklabels([f"d{d}" for d in days])
        ax.set_yticks(range(len(Ws))); ax.set_yticklabels([f"W={W}" for W in Ws])
        ax.set_xlabel("lead day"); ax.set_ylabel("warmup W")
        ax.set_title(var, fontsize=10)
        for wi in range(len(Ws)):
            for di in range(len(days)):
                ax.text(di, wi, f"{mat[wi,di]:+.1f}", ha="center", va="center",
                        fontsize=7, color="black")
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(f"W-sweep heatmap: improvement % (gc_mamba K={cfg.k} vs baseline) — "
                 f"rows = warmup steps, cols = lead day", fontsize=12)
    plt.tight_layout()
    p2 = OUT / "W_sweep_heatmap.png"
    plt.savefig(p2, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p2}")

    # --- lead-mean improvement vs W per variable (line plot) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for var in surface_vars + ["specific_humidity"]:
        ys = []
        for W in Ws:
            pv = data[W]['per_variable_per_lead'][var]
            ys.append(float(np.mean(pv['improvement_gc_mamba_pct'])))
        ax.plot(Ws, ys, "-o", lw=1.5, label=var)
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(alpha=0.3)
    ax.set_xlabel("warmup steps W")
    ax.set_ylabel("lead-mean improvement % vs baseline")
    ax.set_title(f"Lead-mean improvement vs warmup duration W (K={cfg.k})")
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    p3 = OUT / "W_sweep_leadmean_vs_W.png"
    plt.savefig(p3, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p3}")


if __name__ == "__main__":
    main()
