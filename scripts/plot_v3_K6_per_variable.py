#!/usr/bin/env python3
"""Per-variable Z and MSLP train+eval loss plots across K=4 and K=6 phases.

Two views per variable:
  * eval RMSE (latw) ABSOLUTE values — baseline (frozen GraphCast, constant)
    and corrected (frozen + MZ residual). The gap = MZ contribution.
  * eval Δ% (latw) — relative improvement over baseline (paper-comparable).

Plus the train-side per-group loss for MSLP. Z does not have a per-variable
train-loss column (Z is part of `loss_upper` which mixes Z/T/q/u/v/w × 13
levels), so its train trajectory cannot be isolated from the train log.
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


def load_json(p):
    return json.load(open(p))


def get(tl, key, default=np.nan):
    return np.array([e.get(key, default) for e in tl], dtype=float)


def smooth(x, w=20):
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


# ------- load eval logs (K=6 dir contains K=4 chain pre-seeded as well) -----
ev_full = sorted(load_json(K6 / "eval_log.json"), key=lambda e: e["step"])
ev_phase4 = [e for e in ev_full if 5500 < e["step"] <= 7000]
ev_phase6 = [e for e in ev_full if e["step"] > 7000]

# ------- load train log -----------------------------------------------------
tl_full = sorted(load_json(K6 / "train_log.json"), key=lambda e: e["step"])

# ============================================================================
# Plot per-variable: Z and MSLP, 4 panels (2 x 2)
#  A. Z latw RMSE absolute (baseline + corrected)
#  B. Z latw Δ% relative
#  C. MSLP latw RMSE absolute (baseline + corrected)
#  D. MSLP latw Δ% relative
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

def pull_per_var(evs, var):
    steps = np.array([e["step"] for e in evs])
    base = np.array([e.get(f"baseline_{var}_RMSE_latw", np.nan) for e in evs])
    corr = np.array([e.get(f"corrected_{var}_RMSE_latw", np.nan) for e in evs])
    dpct = np.where(base > 0, 100 * (base - corr) / base, np.nan)
    return steps, base, corr, dpct

# Z (geopotential)
s4, b4, c4, d4 = pull_per_var(ev_phase4, "geopotential")
s6, b6, c6, d6 = pull_per_var(ev_phase6, "geopotential")

ax = axes[0, 0]
ax.plot(s4, b4, color="grey", lw=1.5, marker="o", label="baseline (frozen GraphCast, K=4 segs)")
ax.plot(s4, c4, color="C0", lw=1.5, marker="s", label="corrected (MZ + frozen) — K=4 phase")
ax.plot(s6, b6, color="grey", lw=1.5, marker="o", alpha=0.5)
ax.plot(s6, c6, color="C3", lw=1.5, marker="s", label="corrected (MZ + frozen) — K=6 phase")
ax.axvline(7000, color="black", ls=":", alpha=0.5, label="K=4 -> K=6 boundary")
ax.set_xlabel("training step")
ax.set_ylabel("Z (geopotential) RMSE latw (m²/s²)")
ax.set_title("A. Geopotential Z — eval RMSE absolute (baseline vs corrected)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(s4, d4, color="C0", lw=1.5, marker="s",
        label=f"K=4 phase Z Δ% (mean {np.nanmean(d4):+.2f}%)")
ax.plot(s6, d6, color="C3", lw=1.5, marker="s",
        label=f"K=6 phase Z Δ% (mean {np.nanmean(d6):+.2f}%)")
ax.axhline(np.nanmean(d4), color="C0", ls=":", alpha=0.5)
ax.axhline(np.nanmean(d6), color="C3", ls=":", alpha=0.5)
ax.axvline(7000, color="black", ls=":", alpha=0.5)
ax.set_xlabel("training step")
ax.set_ylabel("Z Δ% latw (vs baseline)")
ax.set_title("B. Geopotential Z — eval Δ% relative")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

# MSLP
s4, b4, c4, d4 = pull_per_var(ev_phase4, "mean_sea_level_pressure")
s6, b6, c6, d6 = pull_per_var(ev_phase6, "mean_sea_level_pressure")

ax = axes[1, 0]
ax.plot(s4, b4, color="grey", lw=1.5, marker="o", label="baseline (frozen GraphCast)")
ax.plot(s4, c4, color="C0", lw=1.5, marker="s", label="corrected (MZ + frozen) — K=4 phase")
ax.plot(s6, b6, color="grey", lw=1.5, marker="o", alpha=0.5)
ax.plot(s6, c6, color="C3", lw=1.5, marker="s", label="corrected (MZ + frozen) — K=6 phase")
ax.axvline(7000, color="black", ls=":", alpha=0.5)
ax.set_xlabel("training step")
ax.set_ylabel("MSLP RMSE latw (Pa)")
ax.set_title("C. Mean sea level pressure — eval RMSE absolute (baseline vs corrected)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(s4, d4, color="C0", lw=1.5, marker="s",
        label=f"K=4 phase MSLP Δ% (mean {np.nanmean(d4):+.2f}%)")
ax.plot(s6, d6, color="C3", lw=1.5, marker="s",
        label=f"K=6 phase MSLP Δ% (mean {np.nanmean(d6):+.2f}%)")
ax.axhline(np.nanmean(d4), color="C0", ls=":", alpha=0.5)
ax.axhline(np.nanmean(d6), color="C3", ls=":", alpha=0.5)
ax.axvline(7000, color="black", ls=":", alpha=0.5)
ax.set_xlabel("training step")
ax.set_ylabel("MSLP Δ% latw (vs baseline)")
ax.set_title("D. Mean sea level pressure — eval Δ% relative")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle(
    "Per-variable train+eval trajectory: Z and MSLP. K=4 phase = blue, K=6 phase = red.\n"
    "Z baseline RMSE ~310 m²/s²; corrected drops by 5-7 m²/s² (1.65-2.1% Δ).\n"
    "MSLP baseline RMSE ~245 Pa; corrected drops by 2.5-3.5 Pa (1.0-1.4% Δ).",
    y=1.00, fontsize=11
)
plt.tight_layout()
out1 = OUT / "v3_K6_eval_per_variable_Z_MSLP.png"
plt.savefig(out1, dpi=120, bbox_inches="tight")
print(f"saved: {out1}")
plt.close(fig)

# ============================================================================
# Train-side: state_loss (legacy total) + loss_mslp (K=6 only) + loss_upper
# (K=6 only). Z train loss is not separately recorded — Z is part of the
# upper group (Z + T + q + u + v + w × 13 levels). loss_upper is plotted as
# the closest available proxy.
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(13, 6))
steps = get(tl_full, "step")
state = get(tl_full, "state_loss")
upper = get(tl_full, "loss_upper")    # NaN where not logged (K=4 phase)
mslp = get(tl_full, "loss_mslp")

m_k4 = (steps >= 5500) & (steps <= 7000)
m_k6 = steps > 7000

# state_loss across both phases (legacy formula, comparable)
ax.plot(steps[m_k4], state[m_k4], color="grey", alpha=0.25, lw=0.5)
ax.plot(steps[m_k6], state[m_k6], color="grey", alpha=0.25, lw=0.5,
        label="state_loss (legacy GraphCast-style total, all 83 ch)")
sm4 = smooth(state[m_k4], 20); sm6 = smooth(state[m_k6], 20)
ax.plot(steps[m_k4][:len(sm4)] + 10, sm4, color="grey", lw=2.5,
        label=f"state_loss smoothed K=4 (final {state[m_k4][-200:].mean():.4f})")
ax.plot(steps[m_k6][:len(sm6)] + 10, sm6, color="black", lw=2.5,
        label=f"state_loss smoothed K=6 (final {state[m_k6][-200:].mean():.4f})")

# K=6-only group losses
ax.plot(steps[m_k6], upper[m_k6], color="C0", alpha=0.25, lw=0.5)
sm_u = smooth(upper[m_k6], 20)
ax.plot(steps[m_k6][:len(sm_u)] + 10, sm_u, color="C0", lw=2.5,
        label=f"loss_upper (Z+T+q+u+v+w, 78 ch; K=6 phase only) — final {upper[m_k6][-200:].mean():.4f}")
ax.plot(steps[m_k6], mslp[m_k6], color="C2", alpha=0.25, lw=0.5)
sm_m = smooth(mslp[m_k6], 20)
ax.plot(steps[m_k6][:len(sm_m)] + 10, sm_m, color="C2", lw=2.5,
        label=f"loss_mslp (1 ch, MSLP only; K=6 phase only) — final {mslp[m_k6][-200:].mean():.4f}")

ax.axvline(7000, color="black", ls=":", alpha=0.5)
ax.set_xlabel("training step")
ax.set_ylabel("train loss (lat * lead-weighted normalised MSE)")
ax.set_title(
    "Train-side losses across K=4 + K=6.\n"
    "Z is mixed inside loss_upper (78 channels: Z + T + q + u + v + w × 13 levels) and cannot be isolated.\n"
    "MSLP IS its own group — loss_mslp is the per-variable train loss for MSLP (sits ~10x lower than upper because\n"
    "MSLP is a single channel and pressures are pre-normalised by diff stddev)."
)
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
out2 = OUT / "v3_K6_train_per_group_Z_MSLP.png"
plt.savefig(out2, dpi=120, bbox_inches="tight")
print(f"saved: {out2}")
plt.close(fig)

# Numerical summary
print("\n=== Per-variable summary, K=4 vs K=6 ===")
for var in ("geopotential", "mean_sea_level_pressure"):
    s4, b4, c4, d4 = pull_per_var(ev_phase4, var)
    s6, b6, c6, d6 = pull_per_var(ev_phase6, var)
    name = "Z" if var == "geopotential" else "MSLP"
    print(f"\n{name}:")
    print(f"  K=4 phase (step 5800-7200, n={len(s4)}): "
          f"baseline RMSE={np.nanmean(b4):.4f}  corrected={np.nanmean(c4):.4f}  Δ%={np.nanmean(d4):+.3f}% ± {np.nanstd(d4):.3f}")
    print(f"  K=6 phase (step 7200-current, n={len(s6)}): "
          f"baseline RMSE={np.nanmean(b6):.4f}  corrected={np.nanmean(c6):.4f}  Δ%={np.nanmean(d6):+.3f}% ± {np.nanstd(d6):.3f}")
    print(f"  K=6 - K=4 Δ% gain: {np.nanmean(d6) - np.nanmean(d4):+.3f}pp")
