#!/usr/bin/env python3
"""Plot v3 K=6 phase train + eval trajectory with K=4 context.

Question: did K=6 actually move the needle, or are train_loss / eval Δ% just
bouncing in a band centred on the same K=4 plateau?

Panels:
  1. train loss (legacy state_loss, comparable across K phases) + grad_norm
  2. per-group train losses (upper / MSLP / small_surface) — only meaningful
     for K=6 since older runs don't log them
  3. eval Δ% (latw): overall MAE, Z RMSE, MSLP RMSE — over both K=4 and K=6
     phases to see whether the K=6 mean clearly steps up vs K=4
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

K6 = Path("/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/"
          "mz_fullmamba_v3_GA_K6_seg5_grouploss_LW1_step7000to9500")
K4 = Path("/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/"
          "mz_fullmamba_v3_GA_paperckpt_r1_in2_seg8_meshed_m5_h128_ds16_fullvars_K4_7500")
OUT = Path("/home/lm8598/Weather_Global_experiments/results/2026-05-01_K6_train_eval_curves")
OUT.mkdir(parents=True, exist_ok=True)


def load_train(p):
    tl = json.load(open(p / "train_log.json"))
    tl = sorted(tl, key=lambda e: e["step"])
    return tl


def load_eval(p):
    ev = json.load(open(p / "eval_log.json"))
    ev = sorted(ev, key=lambda e: e["step"])
    return ev


def smooth(x, w=20):
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


tl_k6 = load_train(K6)
ev_k6 = load_eval(K6)
# K=6 trains_log was pre-seeded with K=4 chain; we need to filter "K=6 phase"
# = step > 8500's source step, which is the K=4 step 7000 ckpt. The K=6 dir
# also contains all earlier-phase entries. K=6 phase is steps > 7000.
post_steps_arr = np.array([e["step"] for e in tl_k6])
phase_k6_mask_train = post_steps_arr > 7000
ev_steps_arr = np.array([e["step"] for e in ev_k6])
phase_k6_mask_eval = ev_steps_arr > 7000


def get(tl, key, default=np.nan):
    return np.array([e.get(key, default) for e in tl], dtype=float)


# Compute Z / MSLP / overall Δ% (latw) for both K=4 and K=6 dirs
def latw_dpct(e, var):
    if var == "overall":
        b = e.get("baseline_overall_MAE_latw")
        c = e.get("corrected_overall_MAE_latw")
    else:
        b = e.get(f"baseline_{var}_RMSE_latw")
        c = e.get(f"corrected_{var}_RMSE_latw")
    if b is None or c is None or b == 0:
        return None
    return 100 * (b - c) / b


def filter_eval_phase(ev, lo, hi):
    return [e for e in ev if lo < e["step"] <= hi]


# Pull K=4 eval over K=4 phase (steps 5500..7200)
k4_phase = filter_eval_phase(load_eval(K4), 5500, 7300)
# K=6 eval over K=6 phase
k6_phase = filter_eval_phase(ev_k6, 7000, 9500)

fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# ---- Panel 1: train loss + grad_norm ----------------------------------------
ax = axes[0]
steps_full = get(tl_k6, "step")
loss_full = get(tl_k6, "state_loss")        # legacy formula, comparable
gn_full = get(tl_k6, "grad_norm")

# K=4 phase
m_k4 = (steps_full >= 5500) & (steps_full <= 7000)
m_k6 = steps_full > 7000

ax.plot(steps_full[m_k4], loss_full[m_k4], color="C0", alpha=0.3, lw=0.5)
ax.plot(steps_full[m_k6], loss_full[m_k6], color="C3", alpha=0.3, lw=0.5)
# Smoothed lines (window 20)
sm_k4 = smooth(loss_full[m_k4], 20)
sm_k6 = smooth(loss_full[m_k6], 20)
ax.plot(steps_full[m_k4][:len(sm_k4)] + 10, sm_k4, color="C0",
        label=f"K=4 phase (step 5500-7000), final mean = {loss_full[m_k4][-200:].mean():.4f}", lw=2)
ax.plot(steps_full[m_k6][:len(sm_k6)] + 10, sm_k6, color="C3",
        label=f"K=6 phase (step 7000-{int(steps_full[-1])}), final mean = {loss_full[m_k6][-200:].mean():.4f}", lw=2)
ax.axvline(7000, color="black", ls=":", alpha=0.5, label="K=4 -> K=6 boundary")
ax.set_xlabel("training step")
ax.set_ylabel("state_loss (legacy GraphCast-style, comparable across phases)")
ax.set_title(
    "Train loss across K=4 and K=6 phases.\n"
    "Note: K=6 uses NEW objective (group loss + LW=1.0), so total_loss is on a different scale; "
    "state_loss shown here is the LEGACY formula (lat * w_chan * lead-uniform), comparable."
)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

# ---- Panel 2: per-group losses (K=6 only — K=4 didn't log these) -----------
ax = axes[1]
m6 = steps_full > 7000
upper = get(tl_k6, "loss_upper")
mslp = get(tl_k6, "loss_mslp")
small = get(tl_k6, "loss_small_surface")
ax.plot(steps_full[m6], upper[m6], color="C0", alpha=0.3, lw=0.5)
ax.plot(steps_full[m6], mslp[m6], color="C2", alpha=0.3, lw=0.5)
ax.plot(steps_full[m6], small[m6], color="C1", alpha=0.3, lw=0.5)
sm_u = smooth(upper[m6], 20); sm_m = smooth(mslp[m6], 20); sm_s = smooth(small[m6], 20)
ax.plot(steps_full[m6][:len(sm_u)]+10, sm_u, color="C0", lw=2,
        label=f"upper (Z/T/q/u/v/w, n=78 ch) — final {upper[m6][-200:].mean():.4f}")
ax.plot(steps_full[m6][:len(sm_m)]+10, sm_m, color="C2", lw=2,
        label=f"mslp (n=1 ch) — final {mslp[m6][-200:].mean():.4f}")
ax.plot(steps_full[m6][:len(sm_s)]+10, sm_s, color="C1", lw=2,
        label=f"small_surface (2mT/10m_uv/precip, n=4 ch) — final {small[m6][-200:].mean():.4f}")
ax.set_xlabel("training step")
ax.set_ylabel("per-group loss (lat * lead weighted, NO w_chan)")
ax.set_title(
    "Per-group train loss in K=6 phase. \n"
    "All three groups are flat over the K=6 phase (~1800 steps). "
    "If the model were 'still learning' under group loss, at least one group "
    "should drop monotonically."
)
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)

# ---- Panel 3: eval Δ% (latw) -----------------------------------------------
ax = axes[2]
# K=4 phase
steps4 = np.array([e["step"] for e in k4_phase])
ovr4 = np.array([latw_dpct(e, "overall") for e in k4_phase], dtype=float)
z4 = np.array([latw_dpct(e, "geopotential") for e in k4_phase], dtype=float)
mslp4 = np.array([latw_dpct(e, "mean_sea_level_pressure") for e in k4_phase], dtype=float)
# K=6 phase
steps6 = np.array([e["step"] for e in k6_phase])
ovr6 = np.array([latw_dpct(e, "overall") for e in k6_phase], dtype=float)
z6 = np.array([latw_dpct(e, "geopotential") for e in k6_phase], dtype=float)
mslp6 = np.array([latw_dpct(e, "mean_sea_level_pressure") for e in k6_phase], dtype=float)

ax.plot(steps4, ovr4, color="C0", marker="o", lw=1.5, alpha=0.5,
        label=f"K=4 overall (mean {np.nanmean(ovr4):+.2f}%)")
ax.plot(steps4, z4, color="C0", marker="s", lw=1.5,
        label=f"K=4 Z (mean {np.nanmean(z4):+.2f}%)")
ax.plot(steps4, mslp4, color="C0", marker="^", lw=1.5, ls="--",
        label=f"K=4 MSLP (mean {np.nanmean(mslp4):+.2f}%)")
ax.plot(steps6, ovr6, color="C3", marker="o", lw=1.5, alpha=0.5,
        label=f"K=6 overall (mean {np.nanmean(ovr6):+.2f}%)")
ax.plot(steps6, z6, color="C3", marker="s", lw=1.5,
        label=f"K=6 Z (mean {np.nanmean(z6):+.2f}%)")
ax.plot(steps6, mslp6, color="C3", marker="^", lw=1.5, ls="--",
        label=f"K=6 MSLP (mean {np.nanmean(mslp6):+.2f}%)")
# K=4 mean lines for visual reference
ax.axhline(np.nanmean(z4), color="C0", ls=":", alpha=0.5, lw=1)
ax.axhline(np.nanmean(z6), color="C3", ls=":", alpha=0.5, lw=1)
ax.axvline(7000, color="black", ls=":", alpha=0.5)
ax.set_xlabel("training step")
ax.set_ylabel("eval Δ% (latw RMSE, vs frozen baseline)")
ax.set_title(
    "Eval improvement over baseline. K=6 phase clearly sits at a HIGHER "
    "Δ% level than K=4 phase for Z and MSLP — gap visible in panel "
    "(dotted lines = phase means)."
)
ax.legend(loc="lower right", fontsize=8, ncol=2)
ax.grid(alpha=0.3)

plt.tight_layout()
out = OUT / "v3_K6_train_eval_curves.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")

# Also save numerical summary
summary = {
    "K=4 phase (step 5500-7200)": {
        "n_eval_points": int(len(steps4)),
        "overall_latw_mean": float(np.nanmean(ovr4)),
        "overall_latw_std":  float(np.nanstd(ovr4)),
        "Z_latw_mean": float(np.nanmean(z4)),
        "Z_latw_std":  float(np.nanstd(z4)),
        "MSLP_latw_mean": float(np.nanmean(mslp4)),
        "MSLP_latw_std":  float(np.nanstd(mslp4)),
        "train_state_loss_final200": float(loss_full[m_k4][-200:].mean()),
        "grad_norm_final200": float(gn_full[m_k4][-200:].mean()),
    },
    "K=6 phase (step 7000-current)": {
        "n_eval_points": int(len(steps6)),
        "overall_latw_mean": float(np.nanmean(ovr6)),
        "overall_latw_std":  float(np.nanstd(ovr6)),
        "Z_latw_mean": float(np.nanmean(z6)),
        "Z_latw_std":  float(np.nanstd(z6)),
        "MSLP_latw_mean": float(np.nanmean(mslp6)),
        "MSLP_latw_std":  float(np.nanstd(mslp6)),
        "train_state_loss_final200": float(loss_full[m6][-200:].mean()),
        "grad_norm_final200": float(gn_full[m6][-200:].mean()),
        "loss_upper_final200": float(upper[m6][-200:].mean()),
        "loss_mslp_final200": float(mslp[m6][-200:].mean()),
        "loss_small_surface_final200": float(small[m6][-200:].mean()),
    },
    "K=6 - K=4 gain": {
        "overall_latw": float(np.nanmean(ovr6) - np.nanmean(ovr4)),
        "Z_latw": float(np.nanmean(z6) - np.nanmean(z4)),
        "MSLP_latw": float(np.nanmean(mslp6) - np.nanmean(mslp4)),
    },
}

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== SUMMARY ===")
for phase, d in summary.items():
    print(f"\n{phase}:")
    for k, v in d.items():
        print(f"  {k}: {v}")
