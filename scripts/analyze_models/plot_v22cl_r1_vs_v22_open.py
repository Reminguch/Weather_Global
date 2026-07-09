"""Compare v22cl res=1 closed-loop (best-per-lead envelope, cold_full) against
v22 open-loop (published paper eval, cold_bp) — direct overlay on top of the
same 2-panel structure as 2026-05-23-v22/plots/v22_vs_baseline.png.

Left panel: improvement% vs lead time, all K.
Right panel: K-scan by selected leads.
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
V22_OPEN = ROOT / "results/2026-6-26-v22-kscan-zero-init"       # v22 open-loop, zero-init eval
D14 = ROOT / "results/2026-07-01-v22cl-r1-step14k-eval"
D20 = ROOT / "results/2026-6-29-v22cl-r1-step20k-eval"
DSW = ROOT / "results/2026-07-02-v22cl-r1-allK-sweep"
OUT = ROOT / "results/2026-07-02-v22cl-r1-allK-sweep/plots"

KSCAN = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
STEPS = [6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]

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


def mse_imp(ev, k_idx):
    tb = tf = 0.0
    for v in ev["per_variable_per_step"]:
        if v not in SIGMA: continue
        w = W_VAR.get(v, 1.0); s = SIGMA[v]
        rb = ev["per_variable_per_step"][v]["rmse_baseline"][k_idx]
        rf = ev["per_variable_per_step"][v]["rmse_full"][k_idx]
        tb += w * rb**2 / s**2; tf += w * rf**2 / s**2
    return (1 - tf / tb) * 100 if tb > 0 else np.nan


def cl_path(K, step):
    if step == 14000: return D14 / f"v22cl_r1_K{K}_step{step}_cold_full_zero.json"
    if step == 20000: return D20 / f"v22cl_r1_K{K}_step{step}_cold_full_zero.json"
    return DSW / f"v22cl_r1_K{K}_step{step}_cold_full_zero.json"


def open_path(K):
    return V22_OPEN / f"v22_K{K}_K40_zero_init.json"


def load_v22cl_envelope():
    imp_env = {}
    for K in KSCAN:
        stack = []
        for step in STEPS:
            p = cl_path(K, step)
            if not p.exists(): continue
            ev = json.loads(p.read_text())
            stack.append([mse_imp(ev, k) for k in range(ev["target_steps"])])
        if stack:
            arr = np.array(stack)
            imp_env[K] = arr.max(axis=0)   # best-per-lead
    return imp_env


def load_v22_open():
    imp = {}
    for K in KSCAN:
        p = open_path(K)
        if not p.exists(): continue
        ev = json.loads(p.read_text())
        imp[K] = np.array([mse_imp(ev, k) for k in range(ev["target_steps"])])
    return imp


def plot(imp_open, imp_cl):
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    cmap = cm.get_cmap("viridis")
    Ks = KSCAN
    norm = plt.Normalize(min(Ks), max(Ks))

    ax = axes[0]
    for K in Ks:
        if K in imp_open:
            y = imp_open[K]; leads = np.arange(1, len(y)+1)*6
            ax.plot(leads, y, "--", color=cmap(norm(K)), lw=1.4, alpha=0.85,
                    label=f"v22 open K={K}" if K in [2, 8, 14, 22] else None)
        if K in imp_cl:
            y = imp_cl[K]; leads = np.arange(1, len(y)+1)*6
            ax.plot(leads, y, "-o", ms=3, color=cmap(norm(K)), lw=1.8,
                    label=f"v22cl envelope K={K}" if K in [2, 8, 14, 22] else None)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lead time (h)"); ax.set_ylabel("paper-weighted MSE improvement vs GC (%)")
    ax.set_title("v22 open-loop (dashed) vs v22cl res=1 best-per-lead envelope (solid, cold_full)")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8, ncol=2)

    # K-scan by lead
    ax = axes[1]
    SEL = [6, 24, 48, 72, 120, 168, 240]
    cmap2 = cm.get_cmap("tab10")
    for i, lh in enumerate(SEL):
        k = lh // 6 - 1
        y_open = [imp_open[K][k] if K in imp_open and k < len(imp_open[K]) else np.nan for K in Ks]
        y_cl   = [imp_cl[K][k]   if K in imp_cl   and k < len(imp_cl[K])   else np.nan for K in Ks]
        ax.plot(Ks, y_open, "--", ms=5, color=cmap2(i), lw=1.4, alpha=0.7,
                label=f"open {lh}h")
        ax.plot(Ks, y_cl, "-o", ms=6, color=cmap2(i), lw=1.8,
                label=f"cl {lh}h")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("K"); ax.set_ylabel("paper-weighted MSE improvement (%)")
    ax.set_title("K-scan at selected leads — open (dashed) vs closed best-per-lead (solid)")
    ax.set_xticks(Ks); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")

    fig.suptitle("v22cl res=1 closed-loop (best-per-lead envelope, cold_full) vs v22 open-loop paper eval",
                 fontsize=12)
    plt.tight_layout()
    out = OUT / "v22cl_r1_envelope_vs_v22_open.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def print_diff_table(imp_open, imp_cl):
    print(f"\n{'='*100}")
    print("Δ (v22cl best-per-lead envelope) − (v22 open-loop) at selected leads")
    print(f"{'='*100}")
    SEL = [(3, "24h"), (7, "48h"), (11, "72h"), (19, "120h"), (27, "168h"), (39, "240h")]
    header = "  K  " + "  ".join(f"  {lbl:>16s}" for _, lbl in SEL)
    print(header)
    for K in KSCAN:
        if K not in imp_open or K not in imp_cl: continue
        row = f"  K={K:<2d}"
        for idx, _ in SEL:
            o = imp_open[K][idx]; c = imp_cl[K][idx]
            row += f"  {c:+6.2f} vs {o:+6.2f}"
        print(row)


if __name__ == "__main__":
    imp_open = load_v22_open()
    imp_cl = load_v22cl_envelope()
    print_diff_table(imp_open, imp_cl)
    plot(imp_open, imp_cl)
