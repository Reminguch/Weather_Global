"""Compare GC_mamba (gcmamba_v2) vs residual_mamba closed-loop sg, both at res=2,
across all K values and lead times.

Outputs:
  - Console table: imp% by K by lead
  - Plot 1: imp% vs K at multiple leads (lead = 24, 72, 120, 168, 240h)
  - Plot 2: imp% vs lead for selected K values
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

GCMAMBA_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/results/2026-06-16-gcmamba_v2")
CL_DIR      = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-27-v22cl-kscan-eval")
OUT_DIR     = Path("/home/lm8598/Weather_Global_experiments/results/2026-06-28-gcmamba-vs-residual-cl/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def mse_imp_gcmamba(ev, k_idx):
    """GC_mamba eval JSON uses per_variable_per_lead with rmse_gc_mamba key."""
    pv = ev.get("per_variable_per_lead", {})
    tb = tf = 0.0
    for v, vd in pv.items():
        if v not in SIGMA: continue
        w = W_VAR.get(v, 1.0); s = SIGMA[v]
        rb = vd["rmse_baseline"][k_idx]
        rf = vd["rmse_gc_mamba"][k_idx]
        tb += w * rb**2 / s**2; tf += w * rf**2 / s**2
    return (1 - tf / tb) * 100 if tb > 0 else 0


def mse_imp_residual(ev, k_idx):
    """residual_mamba eval JSON uses per_variable_per_step."""
    pv = ev.get("per_variable_per_step", {})
    tb = tf = 0.0
    for v, vd in pv.items():
        if v not in SIGMA: continue
        w = W_VAR.get(v, 1.0); s = SIGMA[v]
        rb = vd["rmse_baseline"][k_idx]
        rf = vd["rmse_full"][k_idx]
        tb += w * rb**2 / s**2; tf += w * rf**2 / s**2
    return (1 - tf / tb) * 100 if tb > 0 else 0


def load_gcmamba_eval(K):
    """gcmamba_v2 eval JSON for K at step varying (K=1 step19000, rest step39000)."""
    if K == 1:
        p = GCMAMBA_DIR / "eval_res2_2way_K1_step19000.json"
    else:
        p = GCMAMBA_DIR / f"eval_res2_2way_K{K}_step39000.json"
    if not p.exists(): return None
    return json.load(p.open())


def load_residual_cl_eval(K, mode):
    """residual_mamba closed-loop eval JSON."""
    p = CL_DIR / f"v22cl_K{K}_step2000_{mode}_zero.json"
    if not p.exists(): return None
    return json.load(p.open())


KS_GCMAMBA = [1, 2, 4, 8, 12, 16, 20]
KS_RESIDUAL = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
KS_OVERLAP = [1, 2, 4, 8, 12, 16, 20]
LEADS = [24, 72, 120, 168, 240]


def print_table():
    print()
    print("=" * 100)
    print("GC_mamba vs residual_mamba cl cold_full at res=2 — paper-weighted MSE imp% by K, lead")
    print("=" * 100)
    print(f"\n## GC_mamba (gcmamba_v2)")
    print(f"  {'K':>3s}  " + "  ".join(f"L{h}h".rjust(8) for h in LEADS))
    for K in KS_GCMAMBA:
        ev = load_gcmamba_eval(K)
        if ev is None: continue
        cells = [mse_imp_gcmamba(ev, h//6 - 1) for h in LEADS]
        print(f"  K={K:<2d}  " + "  ".join(f"{v:+7.2f}%" for v in cells))

    print(f"\n## residual_mamba cl cold_full (v22cl)")
    print(f"  {'K':>3s}  " + "  ".join(f"L{h}h".rjust(8) for h in LEADS))
    for K in KS_RESIDUAL:
        ev = load_residual_cl_eval(K, "cold_full")
        if ev is None: continue
        cells = [mse_imp_residual(ev, h//6 - 1) for h in LEADS]
        print(f"  K={K:<2d}  " + "  ".join(f"{v:+7.2f}%" for v in cells))

    print(f"\n## Δ (residual cl cold_full − GC_mamba) at overlapping K")
    print(f"  {'K':>3s}  " + "  ".join(f"L{h}h".rjust(8) for h in LEADS))
    for K in KS_OVERLAP:
        ev_g = load_gcmamba_eval(K)
        ev_r = load_residual_cl_eval(K, "cold_full")
        if ev_g is None or ev_r is None: continue
        g_cells = [mse_imp_gcmamba(ev_g, h//6 - 1) for h in LEADS]
        r_cells = [mse_imp_residual(ev_r, h//6 - 1) for h in LEADS]
        deltas = [r - g for r, g in zip(r_cells, g_cells)]
        print(f"  K={K:<2d}  " + "  ".join(f"{v:+7.2f}%" for v in deltas))


def plot_imp_vs_K(out_path: Path):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))

    # PANEL 1: imp% vs K at multiple lead times
    cmap_g = cm.get_cmap("Greens"); cmap_r = cm.get_cmap("Purples")
    for ai, (label_prefix, ks, loader, imp_fn, cmap) in enumerate([
        ("GC_mamba", KS_GCMAMBA, load_gcmamba_eval, mse_imp_gcmamba, cmap_g),
        ("residual cl cold_full", KS_RESIDUAL,
         lambda K: load_residual_cl_eval(K, "cold_full"), mse_imp_residual, cmap_r),
    ]):
        for li, lead_h in enumerate(LEADS):
            xs, ys = [], []
            for K in ks:
                ev = loader(K)
                if ev is None: continue
                ki = lead_h // 6 - 1
                xs.append(K); ys.append(imp_fn(ev, ki))
            if xs:
                color = cmap(0.35 + 0.55 * li / (len(LEADS) - 1))
                ls = "-" if ai == 0 else "--"
                marker = "o" if ai == 0 else "s"
                ax[0].plot(xs, ys, ls, marker=marker, ms=5, lw=2.0, color=color,
                          label=f"{label_prefix} @ L{lead_h}h")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("K (training AR tail)"); ax[0].set_ylabel("paper-weighted MSE imp% vs baseline")
    ax[0].set_xticks(sorted(set(KS_GCMAMBA + KS_RESIDUAL)))
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7, ncol=2, loc="lower right")
    ax[0].set_title("imp% vs K (multiple leads, GC_mamba ○ vs residual cl cold_full □)")

    # PANEL 2: imp% vs lead for selected K
    SEL_K_GCMAMBA = [4, 8, 12, 16, 20]
    SEL_K_RESIDUAL = [4, 8, 12, 16, 18, 22]
    cmap_g2 = cm.get_cmap("Greens"); cmap_r2 = cm.get_cmap("Purples")
    n_lead = 40
    lead_h_full = [6 * (i + 1) for i in range(n_lead)]
    for K in SEL_K_GCMAMBA:
        ev = load_gcmamba_eval(K)
        if ev is None: continue
        ys = [mse_imp_gcmamba(ev, k) for k in range(n_lead)]
        col = cmap_g2(0.35 + 0.55 * (K - min(SEL_K_GCMAMBA)) / (max(SEL_K_GCMAMBA) - min(SEL_K_GCMAMBA) + 1e-9))
        ax[1].plot(lead_h_full, ys, "-", lw=1.8, color=col, label=f"GC_mamba K={K}")
    for K in SEL_K_RESIDUAL:
        ev = load_residual_cl_eval(K, "cold_full")
        if ev is None: continue
        ys = [mse_imp_residual(ev, k) for k in range(n_lead)]
        col = cmap_r2(0.35 + 0.55 * (K - min(SEL_K_RESIDUAL)) / (max(SEL_K_RESIDUAL) - min(SEL_K_RESIDUAL) + 1e-9))
        ax[1].plot(lead_h_full, ys, "--", lw=1.8, color=col, label=f"residual cl K={K}")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("lead time (h)"); ax[1].set_ylabel("paper-weighted MSE imp% vs baseline")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=7, ncol=2, loc="lower center")
    ax[1].set_title("imp% vs lead (selected K, GC_mamba solid vs residual cl dashed)")

    fig.suptitle("res=2: GC_mamba (Mamba in GC) vs residual_mamba closed-loop sg (cold_full) — paper-weighted MSE",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    print_table()
    plot_imp_vs_K(OUT_DIR / "gcmamba_vs_residual_cl_res2.png")
