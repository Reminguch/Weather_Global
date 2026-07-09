"""Regenerate v20 K-scan overlay + metric1_all_K_overlay using the OLD 20-lead JSONs
(/scratch/.../0520_v20_eval_K20/) + the corrected paper formula (diffs_stddev_by_level)."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

W_VAR = {
    "10m_u_component_of_wind": 0.1, "10m_v_component_of_wind": 0.1,
    "2m_temperature": 1.0, "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1, "geopotential": 1.0,
    "temperature": 1.0, "u_component_of_wind": 1.0,
    "v_component_of_wind": 1.0, "vertical_velocity": 1.0, "specific_humidity": 1.0,
}
ds = xr.open_dataset("/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats/diffs_stddev_by_level.nc")
SIGMA = {v: float(ds[v].values if ds[v].values.ndim == 0 else ds[v].values.mean())
         for v in W_VAR if v in ds}


def mse_imp(ev, k_idx):
    tb = tf = 0.0
    for v in ev["per_variable_per_step"]:
        if v not in SIGMA: continue
        w = W_VAR.get(v, 1.0)
        s = SIGMA[v]
        rb = ev["per_variable_per_step"][v]["rmse_baseline"][k_idx]
        rf = ev["per_variable_per_step"][v]["rmse_full"][k_idx]
        tb += w * rb**2 / s**2
        tf += w * rf**2 / s**2
    return (1 - tf / tb) * 100 if tb > 0 else 0


JSON_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0520_v20_eval_K20")
OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-05-20-v20/plots")

Ks = [1, 4, 6, 8, 10, 12, 14]
evs = {}
for K in Ks:
    p = JSON_DIR / f"v20_K{K}_K20.json"
    if p.exists():
        evs[K] = json.loads(p.read_text())

n_lead = evs[Ks[0]]["target_steps"]
lead_h = [6 * (i + 1) for i in range(n_lead)]

# --- Plot 1: 2-panel overlay (matches v22_vs_baseline.png style) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
cmap = cm.get_cmap("viridis")
norm = plt.Normalize(min(Ks), max(Ks))
for K in Ks:
    imps = [mse_imp(evs[K], k_idx) for k_idx in range(n_lead)]
    axes[0].plot(lead_h, imps, "-o", ms=4, lw=1.5, color=cmap(norm(K)), label=f"v20 K={K}")
axes[0].axhline(0, color="k", lw=0.5)
axes[0].set_xlabel("lead time (h)")
axes[0].set_ylabel("improvement vs baseline (%)")
axes[0].set_title(f"v20 (chunk=24, d_conv=4) vs GraphCast baseline — lead 6h..{max(lead_h)}h")
axes[0].grid(alpha=0.3)
axes[0].legend(loc="upper left", fontsize=8, ncol=2)

SELECTED_LEADS = [6, 24, 48, 72, 96, 120]
cmap2 = cm.get_cmap("tab10")
for i, lh in enumerate([h for h in SELECTED_LEADS if h <= max(lead_h)]):
    k_idx = lh // 6 - 1
    imps = [mse_imp(evs[K], k_idx) for K in Ks]
    axes[1].plot(Ks, imps, "-o", ms=5, lw=1.8, color=cmap2(i), label=f"lead {lh}h")
axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_xlabel("K (training AR tail)")
axes[1].set_ylabel("improvement (%)")
axes[1].set_title("v20 K-scan")
axes[1].set_xticks(Ks)
axes[1].grid(alpha=0.3)
axes[1].legend(loc="upper left", fontsize=8)

fig.suptitle(f"v20 (chunk=24, d_conv=4) vs GraphCast baseline — paper formula (diffs_stddev), "
             f"lead 6h..{max(lead_h)}h ({max(lead_h)//24} days)", fontsize=12)
plt.tight_layout()
out1 = OUT_DIR / "v20_vs_baseline_K20.png"
plt.savefig(out1, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"saved {out1}")

# --- Plot 2: single panel metric1_all_K_overlay style ---
fig, ax = plt.subplots(1, 1, figsize=(11, 6))
for K in Ks:
    imps = [mse_imp(evs[K], k_idx) for k_idx in range(n_lead)]
    ax.plot(lead_h, imps, "-o", ms=4, lw=1.8, color=cmap(norm(K)), label=f"K={K}")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("lead time (h)")
ax.set_ylabel("Total weighted MSE improvement vs GraphCast baseline (%)")
ax.set_title("v20: Metric 1 — % improvement vs lead time, across training K (paper formula, diffs_stddev)")
ax.set_xticks([6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120])
ax.tick_params(axis="x", labelsize=8)
ax.grid(alpha=0.3)
ax.legend(title="training K", loc="best", fontsize=9)
plt.tight_layout()
out2 = OUT_DIR / "metric1_all_K_overlay.png"
plt.savefig(out2, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"saved {out2}")
