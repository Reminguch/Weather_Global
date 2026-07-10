"""cold_full_vs_baseline-style plot for res=2 v22cl SWA K-scan.

Runs per-config (inner128 or inner1024). Same visual convention as res=1's
plots/cold_full_vs_baseline.png in 2026-07-02-v22cl-r1-allK-EMA-eval/plots/:
2 panels, paper-weighted MSE improvement, one line per K (left, viridis) /
one line per selected lead (right, tab10).

Usage: python plot_v22cl_res2_cold_full.py <inner128|inner1024>
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import xarray as xr

OUT_DIR = Path(
    "/home/lm8598/Weather_Global_experiments/results/"
    "2026-07-08-Kscan-inner128-vs-inner1024/plots"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

D_KSCAN = Path(
    "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res2_closedloop/"
    "Kscan_SWA_eval"
)
# K=18 lives in the arch-screen eval dir (predates K-scan).
K18_PATHS = {
    "inner1024": Path(
        "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res2_closedloop/"
        "arch_eval_SWA/inner1024_SWA_step3k-8k_cold_full_zero.json"),
    "inner128": Path(
        "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res2_closedloop/"
        "arch_eval_SWA/inner128_SWA_step3k-8k_cold_full_zero.json"),
}

KS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

W_VAR = {
    "10m_u_component_of_wind": 0.1, "10m_v_component_of_wind": 0.1,
    "2m_temperature": 1.0, "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1, "geopotential": 1.0,
    "temperature": 1.0, "u_component_of_wind": 1.0,
    "v_component_of_wind": 1.0, "vertical_velocity": 1.0,
    "specific_humidity": 1.0,
}
SIG_DS = xr.open_dataset(
    "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats/"
    "diffs_stddev_by_level.nc"
)
SIGMA = {v: float(SIG_DS[v].values if SIG_DS[v].values.ndim == 0
                  else SIG_DS[v].values.mean())
         for v in W_VAR if v in SIG_DS}


def mse_imp(ev: dict, k_idx: int) -> float:
    tb = tf = 0.0
    pv = ev["per_variable_per_step"]
    for v in pv:
        if v not in SIGMA: continue
        w = W_VAR.get(v, 1.0); s = SIGMA[v]
        rb = pv[v]["rmse_baseline"][k_idx]
        rf = pv[v]["rmse_full"][k_idx]
        tb += w * rb**2 / s**2
        tf += w * rf**2 / s**2
    return (1 - tf / tb) * 100 if tb > 0 else np.nan


def channel_improve_count(ev, k_idx, exclude_precip=True):
    total, positive = 0, 0
    for c, cd in ev.get("per_channel_per_step", {}).items():
        if exclude_precip and "precipitation" in c: continue
        arr = cd.get("improvement_pct_rmse") or cd.get("improvement_pct")
        if arr and k_idx < len(arr):
            total += 1
            if arr[k_idx] > 0: positive += 1
    return positive, total


def path_for(K: int, config: str) -> Path:
    if K == 18:
        return K18_PATHS[config]
    return D_KSCAN / f"K{K}_{config}_SWA_step3k-8k_cold_full_zero.json"


def load(K: int, config: str):
    p = path_for(K, config)
    if not p.exists(): return None
    return json.loads(p.read_text())


def run(config: str):
    Ks_valid, evs = [], {}
    for K in KS:
        ev = load(K, config)
        if ev is None:
            print(f"K={K} {config}: MISSING at {path_for(K, config)}"); continue
        Ks_valid.append(K); evs[K] = ev
    if not Ks_valid: raise SystemExit(f"no data for {config}")

    n_lead = evs[Ks_valid[0]]["target_steps"]
    lead_h = [6 * (i + 1) for i in range(n_lead)]
    SELECTED = [6, 24, 48, 72, 120, 168, 240]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    cmap = cm.get_cmap("viridis")
    norm = plt.Normalize(min(Ks_valid), max(Ks_valid))

    ax = axes[0]
    for K in Ks_valid:
        imps = [mse_imp(evs[K], k) for k in range(n_lead)]
        ax.plot(lead_h, imps, "-o", ms=4, lw=1.5, color=cmap(norm(K)),
                label=f"K={K}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("paper-weighted MSE improvement (%)")
    ax.set_title(f"v22cl res=2 {config} SWA cold_full — improvement vs lead, all K")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="upper left")

    ax = axes[1]
    cmap2 = cm.get_cmap("tab10")
    for i, lh in enumerate(SELECTED):
        k = lh // 6 - 1
        imps = [mse_imp(evs[K], k) for K in Ks_valid]
        ax.plot(Ks_valid, imps, "-o", ms=5, lw=1.8, color=cmap2(i),
                label=f"lead {lh}h")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("K")
    ax.set_ylabel("paper-weighted MSE improvement (%)")
    ax.set_title(f"v22cl res=2 {config} SWA cold_full K-scan by lead")
    ax.set_xticks(Ks_valid); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"v22cl res=2 {config} SWA(step 3k-8k) — COLD_FULL, zero residual-state init",
        fontsize=12
    )
    plt.tight_layout()
    out = OUT_DIR / f"cold_full_vs_baseline_{config}_res2.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")

    # Channel-improve heatmap
    counts = np.zeros((len(Ks_valid), n_lead))
    totals = np.zeros((len(Ks_valid), n_lead))
    for i, K in enumerate(Ks_valid):
        for k in range(n_lead):
            p, t = channel_improve_count(evs[K], k)
            counts[i, k] = p; totals[i, k] = t

    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    tick_leads = [3, 7, 11, 19, 27, 39]
    tick_lbls = ["24h", "48h", "72h", "120h", "168h", "240h"]

    im0 = axes[0].imshow(counts, aspect="auto", cmap="RdYlGn", origin="lower",
                         vmin=0, vmax=totals.max())
    axes[0].set_yticks(range(len(Ks_valid)))
    axes[0].set_yticklabels([f"K={K}" for K in Ks_valid])
    axes[0].set_xticks(tick_leads); axes[0].set_xticklabels(tick_lbls)
    axes[0].set_xlabel("lead time")
    axes[0].set_title(f"# channels with positive improvement — cold_full ({config})")
    plt.colorbar(im0, ax=axes[0], shrink=0.8, label="# improving channels")
    for i in range(len(Ks_valid)):
        for j in [3, 11, 27, 39]:
            axes[0].text(j, i, f"{int(counts[i, j])}", ha="center", va="center",
                         color="black", fontsize=8)

    frac = np.divide(counts, np.maximum(totals, 1))
    im1 = axes[1].imshow(frac, aspect="auto", cmap="RdYlGn", origin="lower",
                         vmin=0, vmax=1)
    axes[1].set_yticks(range(len(Ks_valid)))
    axes[1].set_yticklabels([f"K={K}" for K in Ks_valid])
    axes[1].set_xticks(tick_leads); axes[1].set_xticklabels(tick_lbls)
    axes[1].set_xlabel("lead time")
    axes[1].set_title(f"fraction of channels improving — cold_full ({config})")
    plt.colorbar(im1, ax=axes[1], shrink=0.8, label="fraction")
    for i in range(len(Ks_valid)):
        for j in [3, 11, 27, 39]:
            axes[1].text(j, i, f"{frac[i, j]:.2f}", ha="center", va="center",
                         color="black", fontsize=8)
    fig.suptitle(f"v22cl res=2 {config} SWA — channels improving vs GC baseline (cold_full)",
                 fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / f"channels_improve_cold_full_{config}_res2.png"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")

    print(f"\n{'='*80}")
    print(f"v22cl res=2 {config} SWA cold_full — paper-weighted MSE improvement%")
    print(f"{'='*80}")
    show = [3, 11, 19, 27, 39]; lbl = ["24h", "72h", "120h", "168h", "240h"]
    print(f"  {'K':>3s}  " + "  ".join(f"{l:>8s}" for l in lbl) + "  | @240h chan+")
    rows = []
    for K in Ks_valid:
        ev = evs[K]
        p240, t240 = channel_improve_count(ev, 39)
        vals = [mse_imp(ev, k) for k in show]
        row = f"  K={K:<2d}"
        for v in vals: row += f"  {v:+7.2f}%"
        row += f"  | {p240}/{t240}"
        print(row)
        rows.append((K, *vals, p240, t240))
    return rows


def combined_plot(rows_inner128, rows_inner1024):
    """Overlay inner128 vs inner1024 K-scan curves for selected leads."""
    SELECTED_LEADS = [(1, "24h"), (2, "72h"), (3, "120h"), (4, "168h"), (5, "240h")]
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = cm.get_cmap("tab10")
    for i, (col_idx, lbl) in enumerate(SELECTED_LEADS):
        ks_H = [r[0] for r in rows_inner128]; ys_H = [r[col_idx] for r in rows_inner128]
        ks_I = [r[0] for r in rows_inner1024]; ys_I = [r[col_idx] for r in rows_inner1024]
        ax.plot(ks_H, ys_H, "-o", color=cmap(i), ms=5, lw=1.6,
                label=f"inner128  {lbl}", alpha=0.9)
        ax.plot(ks_I, ys_I, "--s", color=cmap(i), ms=5, lw=1.6,
                label=f"inner1024 {lbl}", alpha=0.9)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("K"); ax.set_ylabel("paper-weighted MSE improvement (%)")
    ax.set_xticks(KS); ax.grid(alpha=0.3)
    ax.set_title("v22cl res=2 SWA cold_full — inner128 (solid) vs inner1024 (dashed)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout()
    out = OUT_DIR / "cold_full_kscan_inner128_vs_inner1024_overlay.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "both":
        r_H = run("inner128")
        r_I = run("inner1024")
        combined_plot(r_H, r_I)
    else:
        run(args[0])
