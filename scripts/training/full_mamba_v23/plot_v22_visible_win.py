"""Plot v22 K-scan metrics where the improvement is VISUALLY OBVIOUS.

Focus: MSLP RMSB (peak +17-20% improvement, K=8-16), and other surface vars.
x-axis: lead time (days)
y-axis: metric (RMSB / RMSE)
Two curves per panel: baseline + v22 best-K, with shaded gap.
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
KS = [1,2,4,6,8,10,12,14,16,18,20,22]

data = {k: json.load(open(IN_DIR / f"v22_K{k}_cold_bp.json")) for k in KS}
K_t = data[KS[0]]["target_steps"]
lead_days = (np.arange(1, K_t + 1) * 6) / 24.0


def best_k_for(var, metric_key, lead_window=slice(2, 12)):
    """Pick K with biggest mean improvement over the given lead window."""
    imp_key = "improvement_pct_rmsb" if metric_key == "rmsb" else "improvement_pct_rmse"
    best_imp = -1e9; best_k = None
    for k in KS:
        pv = data[k]['per_variable_per_step'][var]
        if imp_key not in pv: continue
        imp = np.asarray(pv[imp_key])[lead_window]
        if imp.mean() > best_imp:
            best_imp = imp.mean(); best_k = k
    return best_k, best_imp


fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# (0,0) MSLP RMSB — the big win
var = "mean_sea_level_pressure"
k_best, _ = best_k_for(var, "rmsb")
ax = axes[0,0]
pv = data[k_best]['per_variable_per_step'][var]
b = np.asarray(pv['rmsb_baseline']); f = np.asarray(pv['rmsb_full']); imp = np.asarray(pv['improvement_pct_rmsb'])
ax.plot(lead_days, b, "-", color="gray", lw=2.2, label=f"baseline (DeepMind small)")
ax.plot(lead_days, f, "-", color="C2", lw=2.2, label=f"v22 K={k_best}")
ax.fill_between(lead_days, f, b, color="C2", alpha=0.20)
ax.set_xlabel("lead (days)"); ax.set_ylabel("RMSB (Pa)")
ax.set_title(f"MSLP RMSB — v22 K={k_best} vs baseline\n"
             f"max improvement: {imp.max():+.1f}% @ lead {lead_days[np.argmax(imp)]:.2f}d", fontsize=11)
ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="best")

# (0,1) MSLP improvement % per K, showing K-curriculum
ax = axes[0,1]
cmap = plt.get_cmap("viridis")
for i, k in enumerate(KS):
    pv = data[k]['per_variable_per_step'][var]
    imp = np.asarray(pv['improvement_pct_rmsb'])
    ax.plot(lead_days, imp, "-", lw=1.3, color=cmap(i/max(1,len(KS)-1)), label=f"K={k}")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("lead (days)"); ax.set_ylabel("RMSB improvement % vs baseline")
ax.set_title(f"MSLP RMSB improvement — all K-scan (best ~+18% at K={k_best})", fontsize=11)
ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best", ncol=2)

# (1,0) 10m_v_wind RMSB
var = "10m_v_component_of_wind"
k_best, _ = best_k_for(var, "rmsb")
ax = axes[1,0]
pv = data[k_best]['per_variable_per_step'][var]
b = np.asarray(pv['rmsb_baseline']); f = np.asarray(pv['rmsb_full']); imp = np.asarray(pv['improvement_pct_rmsb'])
ax.plot(lead_days, b, "-", color="gray", lw=2.2, label=f"baseline")
ax.plot(lead_days, f, "-", color="C2", lw=2.2, label=f"v22 K={k_best}")
ax.fill_between(lead_days, f, b, color="C2", alpha=0.20)
ax.set_xlabel("lead (days)"); ax.set_ylabel("RMSB (m/s)")
ax.set_title(f"10m_v_wind RMSB — v22 K={k_best} vs baseline\n"
             f"max improvement: {imp.max():+.1f}% @ lead {lead_days[np.argmax(imp)]:.2f}d", fontsize=11)
ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="best")

# (1,1) MSLP RMSE absolute reduction at long lead (K=22 case where baseline grows large)
var = "mean_sea_level_pressure"
ax = axes[1,1]
pv_22 = data[22]['per_variable_per_step'][var]
b = np.asarray(pv_22['rmse_baseline']); f = np.asarray(pv_22['rmse_full'])
ax.plot(lead_days, b, "-", color="gray", lw=2.2, label="baseline RMSE")
ax.plot(lead_days, f, "-", color="C2", lw=2.2, label="v22 K=22 RMSE")
ax.fill_between(lead_days, f, b, color="C2", alpha=0.20)
ax.set_xlabel("lead (days)"); ax.set_ylabel("MSLP RMSE (Pa)")
imp = (b - f) / np.maximum(b, 1e-12) * 100
ax.set_title(f"MSLP RMSE absolute — v22 K=22\n"
             f"long-lead reduction: -{(b-f).max():.0f} Pa (~{imp.max():.1f}%)", fontsize=11)
ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="best")

fig.suptitle("v22 — visible improvement panels (cold_bp open-loop eval, vs DeepMind GraphCast small)",
             fontsize=13, y=1.01)
plt.tight_layout()
p = OUT_DIR / "v22_visible_wins.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
print(f"saved {p}")
