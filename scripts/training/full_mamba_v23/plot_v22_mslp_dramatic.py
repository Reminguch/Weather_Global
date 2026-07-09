"""Single dramatic figure: MSLP RMSB v22 K=8 vs baseline.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-06-19-v22-rmsb-rerun")
OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-06-19-v22-rmsb-rerun/plots")

# Load K=8 (best for MSLP RMSB)
d = json.load(open(IN_DIR / "v22_K8_cold_bp.json"))
pv = d['per_variable_per_step']['mean_sea_level_pressure']
b = np.asarray(pv['rmsb_baseline'])
f = np.asarray(pv['rmsb_full'])
imp = np.asarray(pv['improvement_pct_rmsb'])
K_t = len(b)
lead_days = (np.arange(1, K_t + 1) * 6) / 24.0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={'height_ratios': [2, 1]})

# Top: absolute RMSB curves
ax1.plot(lead_days, b, "-", color="#888888", lw=3.0, label="baseline (DeepMind small)")
ax1.plot(lead_days, f, "-", color="#2ca02c", lw=3.0, label="v22 K=8 (Mamba residual)")
ax1.fill_between(lead_days, f, b, color="#2ca02c", alpha=0.20)
# annotate peak improvement point
imax = int(np.argmax(imp))
ax1.annotate(f"−{(b[imax]-f[imax]):.1f} Pa (-{imp[imax]:.1f}%)",
             xy=(lead_days[imax], (b[imax]+f[imax])/2),
             xytext=(lead_days[imax]+0.5, b[imax]+1),
             fontsize=14, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
ax1.set_xlabel("lead time (days)", fontsize=14)
ax1.set_ylabel("MSLP RMSB (Pa)", fontsize=14)
ax1.set_title("MSLP systematic bias (RMSB) — v22 K=8 vs DeepMind GraphCast-small baseline",
              fontsize=14, fontweight="bold")
ax1.legend(fontsize=13, loc="lower right")
ax1.grid(alpha=0.3)
ax1.tick_params(labelsize=12)

# Bottom: improvement %
ax2.plot(lead_days, imp, "-", color="#2ca02c", lw=2.5)
ax2.fill_between(lead_days, 0, imp, where=(imp > 0), color="#2ca02c", alpha=0.30, label="v22 better")
ax2.fill_between(lead_days, 0, imp, where=(imp < 0), color="#d62728", alpha=0.30, label="baseline better")
ax2.axhline(0, color="k", lw=0.7)
ax2.set_xlabel("lead time (days)", fontsize=14)
ax2.set_ylabel("RMSB improvement % vs baseline", fontsize=14)
ax2.set_title(f"v22 K=8 improvement: peak +{imp.max():.1f}% @ lead {lead_days[np.argmax(imp)]:.2f}d, lead-mean +{imp.mean():.1f}%",
              fontsize=13)
ax2.grid(alpha=0.3)
ax2.tick_params(labelsize=12)
ax2.legend(fontsize=11, loc="upper right")

plt.tight_layout()
p = OUT_DIR / "v22_mslp_K8_dramatic.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
print(f"saved {p}")
print(f"baseline RMSB lead 1-5: {[round(x,2) for x in b[:5]]}")
print(f"v22 K=8 RMSB lead 1-5:  {[round(x,2) for x in f[:5]]}")
print(f"improvement %:           {[round(x,1) for x in imp[:5]]}")
print(f"lead-mean improvement:   {imp.mean():.1f}%")
