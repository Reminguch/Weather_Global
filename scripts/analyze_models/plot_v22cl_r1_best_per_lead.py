"""For each K, at each lead time, pick the best-performing checkpoint.
Shows the ENVELOPE (max over ckpts) per K.

Uses paper-weighted MSE improvement, cold_full mode.
Data source: sweep + step14k + step20k JSONs (8 ckpts per K).

Outputs:
  1. best_per_lead_envelope.png       — 11 lines (K), y = best-imp% per lead
  2. per_K_all_ckpts_envelope.png    — 11 subplots, each K: all ckpt curves + envelope highlighted
  3. best_step_per_lead_heatmap.png   — heatmap: rows=K, cols=lead, cell=best step
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
D14 = ROOT / "results/2026-07-01-v22cl-r1-step14k-eval"
D20 = ROOT / "results/2026-6-29-v22cl-r1-step20k-eval"
DSW = ROOT / "results/2026-07-02-v22cl-r1-allK-sweep"
OUT = ROOT / "results/2026-07-02-v22cl-r1-allK-sweep/plots"
OUT.mkdir(parents=True, exist_ok=True)

KSCAN = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
STEPS = [6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
MODE = "cold_full"

W_VAR = {
    "10m_u_component_of_wind": 0.1, "10m_v_component_of_wind": 0.1,
    "2m_temperature": 1.0, "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1, "geopotential": 1.0,
    "temperature": 1.0, "u_component_of_wind": 1.0,
    "v_component_of_wind": 1.0, "vertical_velocity": 1.0, "specific_humidity": 1.0,
}
SIG_DS = xr.open_dataset(
    "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats/diffs_stddev_by_level.nc")
SIGMA = {v: float(SIG_DS[v].values if SIG_DS[v].values.ndim == 0
                  else SIG_DS[v].values.mean())
         for v in W_VAR if v in SIG_DS}


def json_path(K, step):
    if step == 14000: return D14 / f"v22cl_r1_K{K}_step{step}_{MODE}_zero.json"
    if step == 20000: return D20 / f"v22cl_r1_K{K}_step{step}_{MODE}_zero.json"
    return DSW / f"v22cl_r1_K{K}_step{step}_{MODE}_zero.json"


def mse_imp(ev, k_idx):
    tb = tf = 0.0
    for v in ev["per_variable_per_step"]:
        if v not in SIGMA: continue
        w = W_VAR.get(v, 1.0); s = SIGMA[v]
        rb = ev["per_variable_per_step"][v]["rmse_baseline"][k_idx]
        rf = ev["per_variable_per_step"][v]["rmse_full"][k_idx]
        tb += w * rb**2 / s**2; tf += w * rf**2 / s**2
    return (1 - tf / tb) * 100 if tb > 0 else np.nan


def load_all():
    """Return imp[K][step] = array[n_lead]."""
    n_lead = 40
    data = {}   # K -> dict[step -> np.array shape (n_lead,)]
    for K in KSCAN:
        d = {}
        for step in STEPS:
            p = json_path(K, step)
            if not p.exists(): continue
            ev = json.loads(p.read_text())
            d[step] = np.array([mse_imp(ev, k) for k in range(ev["target_steps"])])
        data[K] = d
    return data


def per_K_envelope(data):
    """Return imp_env[K] = (n_lead,) best imp; step_env[K] = (n_lead,) best step."""
    imp_env = {}; step_env = {}
    for K, d in data.items():
        if not d: continue
        steps_avail = sorted(d.keys())
        stack = np.stack([d[s] for s in steps_avail], axis=0)   # [n_steps, n_lead]
        best_idx = np.argmax(stack, axis=0)                     # [n_lead]
        imp_env[K] = stack[best_idx, np.arange(stack.shape[1])]
        step_env[K] = np.array([steps_avail[i] for i in best_idx])
    return imp_env, step_env


def plot_envelope(imp_env, step_env):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    cmap = cm.get_cmap("viridis")
    norm = plt.Normalize(min(KSCAN), max(KSCAN))
    for K in KSCAN:
        if K not in imp_env: continue
        y = imp_env[K]
        leads = np.arange(1, len(y)+1) * 6
        ax.plot(leads, y, "-o", ms=4, lw=1.8, color=cmap(norm(K)), label=f"K={K}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("paper-weighted MSE improvement over GC (%)")
    ax.set_title(f"v22cl res=1 {MODE} — BEST-per-lead envelope across 8 ckpts per K")
    ax.grid(alpha=0.3); ax.legend(ncol=2, fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(OUT / "best_per_lead_envelope.png", dpi=140, bbox_inches="tight")
    print(f"saved {OUT/'best_per_lead_envelope.png'}")
    plt.close(fig)


def plot_per_K_facet(data, imp_env, step_env):
    Ks = [K for K in KSCAN if K in data and data[K]]
    n = len(Ks); ncol = 4; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8*ncol, 3.6*nrow), sharex=True)
    axes = np.atleast_2d(axes).flatten()
    step_cmap = cm.get_cmap("plasma")
    step_norm = plt.Normalize(6000, 20000)
    for i, K in enumerate(Ks):
        ax = axes[i]
        d = data[K]
        steps_avail = sorted(d.keys())
        for s in steps_avail:
            y = d[s]
            leads = np.arange(1, len(y)+1) * 6
            ax.plot(leads, y, "-", color=step_cmap(step_norm(s)), lw=0.9, alpha=0.55,
                    label=f"{s//1000}k")
        # envelope
        y_env = imp_env[K]
        leads = np.arange(1, len(y_env)+1) * 6
        ax.plot(leads, y_env, "-", color="red", lw=2.5, label="envelope")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"K={K}", fontsize=11)
        ax.grid(alpha=0.3)
        if i % ncol == 0: ax.set_ylabel("MSE imp% vs GC")
        if i >= (nrow-1)*ncol: ax.set_xlabel("lead time (h)")
        if i == 0: ax.legend(fontsize=7, ncol=2, loc="lower left")
    for i in range(len(Ks), len(axes)):
        axes[i].axis("off")
    fig.suptitle(f"v22cl res=1 {MODE} — all ckpts per K + red envelope (best-per-lead)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT / "per_K_all_ckpts_envelope.png", dpi=140, bbox_inches="tight")
    print(f"saved {OUT/'per_K_all_ckpts_envelope.png'}")
    plt.close(fig)


def plot_best_step_heatmap(step_env):
    n_lead = 40
    Ks_valid = [K for K in KSCAN if K in step_env]
    mat = np.zeros((len(Ks_valid), n_lead), dtype=int)
    for i, K in enumerate(Ks_valid):
        mat[i, :] = step_env[K]
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(mat, aspect="auto", cmap="plasma", origin="lower",
                   vmin=6000, vmax=20000)
    ax.set_yticks(np.arange(len(Ks_valid)))
    ax.set_yticklabels([f"K={K}" for K in Ks_valid])
    lead_ticks_h = [24, 48, 72, 120, 168, 240]
    ax.set_xticks([h//6 - 1 for h in lead_ticks_h])
    ax.set_xticklabels([f"{h}h" for h in lead_ticks_h])
    ax.set_xlabel("lead time")
    ax.set_title("best ckpt per (K, lead) — cell = training step number that maximizes cold_full MSE imp%")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="training step")
    # Annotate step number
    for i in range(mat.shape[0]):
        for j in range(0, mat.shape[1], 4):
            ax.text(j, i, f"{mat[i,j]//1000}k", ha="center", va="center",
                    color="white" if mat[i,j] > 14000 else "black", fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "best_step_per_lead_heatmap.png", dpi=140, bbox_inches="tight")
    print(f"saved {OUT/'best_step_per_lead_heatmap.png'}")
    plt.close(fig)


def print_summary(imp_env, step_env):
    print(f"\n{'='*90}")
    print("Best-per-lead envelope: imp% (step) at selected leads")
    print(f"{'='*90}")
    sel = [(3, "24h"), (7, "48h"), (11, "72h"), (19, "120h"), (27, "168h"), (39, "240h")]
    header = "  K   " + "  ".join(f"{lbl:>14s}" for _, lbl in sel)
    print(header)
    for K in KSCAN:
        if K not in imp_env: continue
        row = f"  K={K:<2d}"
        for idx, lbl in sel:
            imp = imp_env[K][idx]; step = step_env[K][idx]
            row += f"  {imp:>+6.2f}% ({step//1000}k)"
        print(row)


if __name__ == "__main__":
    data = load_all()
    imp_env, step_env = per_K_envelope(data)
    print_summary(imp_env, step_env)
    plot_envelope(imp_env, step_env)
    plot_per_K_facet(data, imp_env, step_env)
    plot_best_step_heatmap(step_env)
