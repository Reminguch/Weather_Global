"""Data supporting the SWA rationale in the closed_loop_v22cl README.

Produces three artifacts:
  1. per-step improvement% at lead 240h vs training step for K=14/18/20/22
     (shows the characteristic peak-then-decay pattern)
  2. per-K table: best-single-step vs step 20k vs SWA improvement%
  3. K=22 alpha-sweep: step 6k / step 20k × alpha={0.25, 0.5, 0.75, 1.0}
     — shows step 20k with alpha<1 recovers most of the loss
     (direct evidence that residual amplitude is the cause of drift)
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ROOT = Path("/home/lm8598/Weather_Global_experiments/results")
SWEEP = ROOT / "2026-07-02-v22cl-r1-allK-sweep"
EMA   = ROOT / "2026-07-02-v22cl-r1-allK-EMA-eval"
STEP20K = ROOT / "2026-6-29-v22cl-r1-step20k-eval"
ALPHA = ROOT / "2026-07-02-v22cl-K22-alpha-sweep"
OUT   = Path("/tmp/closed-loop-push-wt/closed_loop_v22cl/results/SWA_full_eval/plots")
OUT.mkdir(parents=True, exist_ok=True)

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


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


# =========================================================================
# 1. per-step improvement% at lead 240h vs training step, per K
# =========================================================================
STEPS_SWEEP = [6000, 8000, 10000, 12000, 16000, 18000]
KS_SHOW = [14, 18, 20, 22]
K_240 = 39  # lead index for 240h (K=40 rollout, 0-indexed)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
colors = {14: "#4c72b0", 18: "#55a868", 20: "#c44e52", 22: "#8172b2"}
for K in KS_SHOW:
    steps, imps = [], []
    for s in STEPS_SWEEP:
        ev = load(SWEEP / f"v22cl_r1_K{K}_step{s}_cold_full_zero.json")
        if ev is None: continue
        steps.append(s); imps.append(mse_imp(ev, K_240))
    ev20 = load(STEP20K / f"v22cl_r1_K{K}_step20000_cold_full_zero.json")
    if ev20 is not None:
        steps.append(20000); imps.append(mse_imp(ev20, K_240))
    if steps:
        ax.plot(steps, imps, "-o", ms=6, lw=1.8, color=colors[K], label=f"K={K}")
        # Overlay SWA horizontal line
        ev_swa = load(EMA / f"v22cl_r1_K{K}_EMA_cold_full_zero.json")
        if ev_swa is not None:
            swa_imp = mse_imp(ev_swa, K_240)
            ax.axhline(swa_imp, ls="--", lw=1.2, color=colors[K], alpha=0.6,
                       label=f"K={K} SWA (imp={swa_imp:+.2f}%)")

ax.set_xlabel("training step")
ax.set_ylabel("paper-weighted MSE improvement% @ 240h (cold_full)")
ax.set_title("Post-optimum drift: per-K single-step improvement decays after peak;\n"
             "SWA (dashed) beats every single step")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2, loc="lower left")
plt.tight_layout()
plt.savefig(OUT / "drift_vs_step_lead240.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"saved {OUT / 'drift_vs_step_lead240.png'}")


# =========================================================================
# 2. per-K table: best-step vs step 20k vs SWA @ 240h + mean(24-240h)
# =========================================================================
rows = []
for K in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]:
    # Gather all sweep steps + step20k + SWA
    step_imps = {}
    for s in STEPS_SWEEP:
        ev = load(SWEEP / f"v22cl_r1_K{K}_step{s}_cold_full_zero.json")
        if ev is not None:
            step_imps[s] = mse_imp(ev, K_240)
    ev20 = load(STEP20K / f"v22cl_r1_K{K}_step20000_cold_full_zero.json")
    if ev20 is not None:
        step_imps[20000] = mse_imp(ev20, K_240)
    if not step_imps: continue
    best_step = max(step_imps, key=step_imps.get)
    best_val = step_imps[best_step]
    ev_swa = load(EMA / f"v22cl_r1_K{K}_EMA_cold_full_zero.json")
    swa_val = mse_imp(ev_swa, K_240) if ev_swa else np.nan
    step20_val = step_imps.get(20000, np.nan)
    step6_val = step_imps.get(6000, np.nan)
    rows.append((K, step6_val, best_step, best_val, step20_val, swa_val))

with open(OUT / "swa_vs_single_ckpts_table.csv", "w") as f:
    f.write("K,step6k_imp%,best_step,best_step_imp%,step20k_imp%,SWA_imp%,SWA-best_gain%\n")
    for K, s6, bs, bv, s20, sw in rows:
        gain = sw - bv if not np.isnan(sw) else np.nan
        f.write(f"{K},{s6:.2f},{bs},{bv:.2f},{s20:.2f},{sw:.2f},{gain:+.2f}\n")
print(f"saved {OUT / 'swa_vs_single_ckpts_table.csv'}")

# Companion bar plot
Ks = [r[0] for r in rows]
step6 = [r[1] for r in rows]
best = [r[3] for r in rows]
step20 = [r[4] for r in rows]
swa = [r[5] for r in rows]
fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
x = np.arange(len(Ks)); w = 0.22
ax.bar(x - 1.5*w, step6, w, label="step 6k (early)",   color="#a4c2f4")
ax.bar(x - 0.5*w, best,  w, label="best single step",   color="#4c72b0")
ax.bar(x + 0.5*w, step20, w, label="step 20k (late)",  color="#f4b183")
ax.bar(x + 1.5*w, swa,   w, label="SWA (avg 6k–16k)",  color="#c44e52")
ax.set_xticks(x); ax.set_xticklabels([f"K={k}" for k in Ks])
ax.set_ylabel("paper-weighted MSE improvement% @ 240h")
ax.set_title("SWA vs single training-step ckpts (cold_full, lead 240h)")
ax.axhline(0, color="k", lw=0.5)
ax.grid(alpha=0.3, axis="y")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "swa_vs_single_ckpts_bar.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"saved {OUT / 'swa_vs_single_ckpts_bar.png'}")


# =========================================================================
# 3. K=22 alpha-sweep: how much of the drift is amplitude?
# =========================================================================
ALPHAS = [0.25, 0.5, 0.75, 1.0]
ALPHA_STR = {0.25: "0p25", 0.5: "0p5", 0.75: "0p75", 1.0: "1p0"}
fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
for label_step, base_step in [("step 6k (near-optimum)", 6000), ("step 20k (overtrained)", 20000)]:
    ys = []
    for a in ALPHAS:
        ev = load(ALPHA / f"v22cl_r1_K22_step{base_step}_cold_full_alpha{ALPHA_STR[a]}_zero.json")
        ys.append(mse_imp(ev, K_240) if ev is not None else np.nan)
    ax.plot(ALPHAS, ys, "-o", ms=8, lw=2, label=label_step)

# Overlay SWA horizontal line
ev_swa = load(EMA / "v22cl_r1_K22_EMA_cold_full_zero.json")
swa_imp = mse_imp(ev_swa, K_240)
ax.axhline(swa_imp, ls="--", lw=1.2, color="C3", label=f"K=22 SWA (imp={swa_imp:+.2f}%)")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel(r"residual amplitude scale $\alpha$  (full_pred = bp + $\alpha$ · rp)")
ax.set_ylabel("paper-weighted MSE improvement% @ 240h")
ax.set_title("K=22 α-sweep: scaling residual amplitude at inference\n"
             "step 20k α=0.5 recovers most of the drift → amplitude IS the cause")
ax.grid(alpha=0.3); ax.legend(loc="lower center", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "K22_alpha_sweep.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"saved {OUT / 'K22_alpha_sweep.png'}")

print("\nDone.")
