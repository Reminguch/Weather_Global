"""Paper-weighted MSE improvement vs training checkpoint (ckpt / epoch), per K.

For each K, we have up to 8 cold_full eval points across training:
  step 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000
    (14000 from step14k eval dir; 20000 from step20k eval dir;
     others from allK-sweep dir)

Plots:
  1. paper_mse_at_240h_vs_ckpt.png — one line per K, y = imp% at 240h
  2. paper_mse_meanleads_vs_ckpt.png — y = mean over leads 24..240h
  3. per_K_lead_facet.png — 11 subplots (one per K), each with imp% at 6
     leads (24/48/72/120/168/240h) vs ckpt
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

ROOT = Path("/home/lm8598/Weather_Global_experiments")
D_STEP14K = ROOT / "results/2026-07-01-v22cl-r1-step14k-eval"
D_STEP20K = ROOT / "results/2026-6-29-v22cl-r1-step20k-eval"
D_SWEEP   = ROOT / "results/2026-07-02-v22cl-r1-allK-sweep"
OUT_DIR   = ROOT / "results/2026-07-02-v22cl-r1-allK-sweep/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KSCAN = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
STEPS = [6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
MODE  = "cold_full"

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


def json_path(K, step):
    if step == 14000:
        return D_STEP14K / f"v22cl_r1_K{K}_step{step}_{MODE}_zero.json"
    if step == 20000:
        return D_STEP20K / f"v22cl_r1_K{K}_step{step}_{MODE}_zero.json"
    return D_SWEEP / f"v22cl_r1_K{K}_step{step}_{MODE}_zero.json"


def mse_imp(ev, k_idx):
    tb = tf = 0.0
    for v in ev["per_variable_per_step"]:
        if v not in SIGMA: continue
        w = W_VAR.get(v, 1.0); s = SIGMA[v]
        rb = ev["per_variable_per_step"][v]["rmse_baseline"][k_idx]
        rf = ev["per_variable_per_step"][v]["rmse_full"][k_idx]
        tb += w * rb**2 / s**2; tf += w * rf**2 / s**2
    return (1 - tf / tb) * 100 if tb > 0 else None


def load_curves():
    """Return dict K -> (steps_arr, imp_matrix [n_step, n_lead])."""
    out = {}
    for K in KSCAN:
        pts_step = []; pts_imp = []
        for step in STEPS:
            p = json_path(K, step)
            if not p.exists(): continue
            ev = json.loads(p.read_text())
            n_lead = ev["target_steps"]
            imps = [mse_imp(ev, k) for k in range(n_lead)]
            pts_step.append(step); pts_imp.append(imps)
        if pts_step:
            out[K] = (np.array(pts_step), np.array(pts_imp))
    return out


def plot_at_lead(curves, lead_idx, lead_label, out_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = cm.get_cmap("viridis")
    Ks_sorted = sorted(curves.keys())
    norm = plt.Normalize(min(Ks_sorted), max(Ks_sorted))
    for K in Ks_sorted:
        s, imp = curves[K]
        y = imp[:, lead_idx]
        ax.plot(s, y, "-o", ms=6, lw=2, color=cmap(norm(K)), label=f"K={K}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("training step (ckpt / epoch)")
    ax.set_ylabel(f"paper-weighted MSE improvement vs GC (%) @ {lead_label}")
    ax.set_title(f"v22cl res=1 {MODE} — improvement@{lead_label} vs training step, all K")
    ax.grid(alpha=0.3); ax.legend(ncol=2, loc="best", fontsize=9)
    ax.set_xticks(STEPS)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out_path}")


def plot_mean_leads(curves, start_idx, end_idx, out_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = cm.get_cmap("viridis")
    Ks_sorted = sorted(curves.keys())
    norm = plt.Normalize(min(Ks_sorted), max(Ks_sorted))
    for K in Ks_sorted:
        s, imp = curves[K]
        y = imp[:, start_idx:end_idx].mean(axis=1)
        ax.plot(s, y, "-o", ms=6, lw=2, color=cmap(norm(K)), label=f"K={K}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("training step (ckpt / epoch)")
    ax.set_ylabel(f"paper-weighted MSE improvement vs GC (%) mean lead {(start_idx+1)*6}h..{end_idx*6}h")
    ax.set_title(f"v22cl res=1 {MODE} — mean improvement vs training step, all K")
    ax.grid(alpha=0.3); ax.legend(ncol=2, loc="best", fontsize=9)
    ax.set_xticks(STEPS)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out_path}")


def plot_per_K_facet(curves, out_path):
    Ks_sorted = sorted(curves.keys())
    n = len(Ks_sorted)
    ncol = 4; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5*ncol, 3.5*nrow), sharex=True)
    axes = np.atleast_2d(axes).flatten()
    SEL_LEAD = [3, 7, 11, 19, 27, 39]   # 24, 48, 72, 120, 168, 240 h
    SEL_LBL  = ["24h", "48h", "72h", "120h", "168h", "240h"]
    cmap = cm.get_cmap("tab10")
    for i, K in enumerate(Ks_sorted):
        ax = axes[i]
        s, imp = curves[K]
        for j, (lidx, lbl) in enumerate(zip(SEL_LEAD, SEL_LBL)):
            y = imp[:, lidx]
            ax.plot(s, y, "-o", ms=5, lw=1.5, color=cmap(j), label=lbl)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"K={K}", fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xticks(STEPS)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        if i % ncol == 0:
            ax.set_ylabel("paper-MSE imp% vs GC")
        if i >= (nrow-1)*ncol:
            ax.set_xlabel("training step (ckpt / epoch)")
        if i == 0:
            ax.legend(fontsize=8, ncol=2, loc="upper left")
    for i in range(len(Ks_sorted), len(axes)):
        axes[i].axis("off")
    fig.suptitle(f"v22cl res=1 {MODE} — paper-weighted MSE imp vs training step, per K", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out_path}")


def main():
    curves = load_curves()
    print(f"K → n_steps loaded:")
    for K in sorted(curves): print(f"  K={K}  n={len(curves[K][0])}  steps={curves[K][0].tolist()}")

    if not curves:
        print("no data yet"); return

    plot_at_lead(curves, 39, "240h", OUT_DIR / "paper_mse_at_240h_vs_ckpt.png")
    plot_at_lead(curves, 19, "120h", OUT_DIR / "paper_mse_at_120h_vs_ckpt.png")
    plot_at_lead(curves, 11, "72h",  OUT_DIR / "paper_mse_at_72h_vs_ckpt.png")
    plot_mean_leads(curves, 3, 40, OUT_DIR / "paper_mse_meanleads_vs_ckpt.png")
    plot_per_K_facet(curves, OUT_DIR / "per_K_lead_facet.png")


if __name__ == "__main__":
    main()
