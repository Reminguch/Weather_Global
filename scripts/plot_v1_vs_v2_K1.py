#!/usr/bin/env python3
"""Compare v1 K=1 best vs v2 K=1 (TIMEOUT'd at step 2200) — train loss and
eval Δ% overlaid, to see whether v2 is just converging more slowly (collapses
onto v1 curve) or plateauing at a worse level."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

R = Path("/home/lm8598/Weather_Global_experiments/results/mz_residual_memory")
V1 = R / "mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_4k_cont"
V2 = R / "mz_fullmamba_v2_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K1_4k"
OUT = Path("/home/lm8598/Weather_Global_experiments/results/2026-04-27_train_eval_curves")
OUT.mkdir(parents=True, exist_ok=True)


def smooth(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def load(run_dir):
    tr = json.load(open(run_dir / "train_log.json"))
    tr.sort(key=lambda e: e["step"])
    ev = json.load(open(run_dir / "eval_log.json"))
    ev.sort(key=lambda e: e["step"])
    return tr, ev


tr1, ev1 = load(V1)
tr2, ev2 = load(V2)

steps1 = np.array([e["step"] for e in tr1])
loss1 = np.array([e.get("loss", e.get("total_loss", 0)) for e in tr1])
steps2 = np.array([e["step"] for e in tr2])
loss2 = np.array([e.get("loss", e.get("total_loss", 0)) for e in tr2])

ev1_steps = np.array([e["step"] for e in ev1])
ev1_dpct = np.array([100 * (e["baseline_overall_MAE"] - e["corrected_overall_MAE"]) / e["baseline_overall_MAE"] for e in ev1])
ev1_z = np.array([100 * (e["baseline_geopotential_RMSE"] - e["corrected_geopotential_RMSE"]) / e["baseline_geopotential_RMSE"] for e in ev1])
ev1_mslp = np.array([100 * (e["baseline_mean_sea_level_pressure_RMSE"] - e["corrected_mean_sea_level_pressure_RMSE"]) / e["baseline_mean_sea_level_pressure_RMSE"] for e in ev1])

ev2_steps = np.array([e["step"] for e in ev2])
ev2_dpct = np.array([100 * (e["baseline_overall_MAE"] - e["corrected_overall_MAE"]) / e["baseline_overall_MAE"] for e in ev2])
ev2_z = np.array([100 * (e["baseline_geopotential_RMSE"] - e["corrected_geopotential_RMSE"]) / e["baseline_geopotential_RMSE"] for e in ev2])
ev2_mslp = np.array([100 * (e["baseline_mean_sea_level_pressure_RMSE"] - e["corrected_mean_sea_level_pressure_RMSE"]) / e["baseline_mean_sea_level_pressure_RMSE"] for e in ev2])

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Train loss MA-200
ax = axes[0, 0]
sm1 = smooth(loss1, 200); sm1_steps = steps1[199:]
sm2 = smooth(loss2, 200); sm2_steps = steps2[199:]
ax.plot(sm1_steps, sm1, color="C0", linewidth=2.5, label="v1 K=1 (full 4000 steps)")
ax.plot(sm2_steps, sm2, color="C3", linewidth=2.5, label="v2 K=1 (TIMEOUT @ 2200)")
ax.set_xlabel("Training step")
ax.set_ylabel("Train loss (MA-200, normalized MSE)")
ax.set_title("Train loss")
ax.grid(alpha=0.3); ax.legend(loc="upper right")

# Eval Δ% overall MAE
ax = axes[0, 1]
ax.plot(ev1_steps, ev1_dpct, color="C0", linewidth=2.5, marker="o", markersize=5, label="v1 K=1")
ax.plot(ev2_steps, ev2_dpct, color="C3", linewidth=2.5, marker="s", markersize=5, label="v2 K=1")
ax.set_xlabel("Training step")
ax.set_ylabel("Eval Δ% (overall MAE, validation 2022, TF mode)")
ax.set_title("Eval improvement (overall MAE Δ%)")
ax.grid(alpha=0.3); ax.legend(loc="lower right")

# Eval Z RMSE Δ%
ax = axes[1, 0]
ax.plot(ev1_steps, ev1_z, color="C0", linewidth=2.5, marker="o", markersize=5, label="v1 K=1")
ax.plot(ev2_steps, ev2_z, color="C3", linewidth=2.5, marker="s", markersize=5, label="v2 K=1")
ax.set_xlabel("Training step")
ax.set_ylabel("Z (geopotential 13L) RMSE Δ%")
ax.set_title("Z RMSE improvement")
ax.grid(alpha=0.3); ax.legend(loc="lower right")

# Eval MSLP RMSE Δ%
ax = axes[1, 1]
ax.plot(ev1_steps, ev1_mslp, color="C0", linewidth=2.5, marker="o", markersize=5, label="v1 K=1")
ax.plot(ev2_steps, ev2_mslp, color="C3", linewidth=2.5, marker="s", markersize=5, label="v2 K=1")
ax.set_xlabel("Training step")
ax.set_ylabel("MSLP RMSE Δ%")
ax.set_title("MSLP improvement")
ax.grid(alpha=0.3); ax.legend(loc="lower right")

plt.suptitle(
    "v1 K=1 best vs v2 K=1 (--specialist-heads --anchor-as-batch)\n"
    "Train loss MA-200 + eval Δ% trajectories. v2 TIMEOUT'd at step 2200.",
    fontsize=11, y=1.005,
)
plt.tight_layout()
out = OUT / "v1_vs_v2_K1.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved: {out}")
plt.close(fig)


# Also print numerical summary
print(f"\n--- Train loss MA-200 ---")
print(f"v1 step 1000: {sm1[np.argmin(np.abs(sm1_steps - 1000))]:.5f}")
print(f"v2 step 1000: {sm2[np.argmin(np.abs(sm2_steps - 1000))]:.5f}")
print(f"v1 step 2000: {sm1[np.argmin(np.abs(sm1_steps - 2000))]:.5f}")
print(f"v2 step 2000: {sm2[np.argmin(np.abs(sm2_steps - 2000))]:.5f}")
print(f"v1 step 4000 (final): {sm1[-1]:.5f}")
print(f"v2 step 2200 (TIMEOUT): {sm2[-1]:.5f}")

print(f"\n--- Eval Δ% (overall MAE) at matched steps ---")
print(f"{'step':>5}  {'v1':>8}  {'v2':>8}  {'gap':>8}")
for s in [200, 1000, 2000, 2200]:
    if s in ev1_steps and s in ev2_steps:
        i1 = int(np.where(ev1_steps == s)[0][0])
        i2 = int(np.where(ev2_steps == s)[0][0])
        gap = ev2_dpct[i2] - ev1_dpct[i1]
        print(f"{s:>5}  {ev1_dpct[i1]:+7.3f}%  {ev2_dpct[i2]:+7.3f}%  {gap:+7.3f}pp")
