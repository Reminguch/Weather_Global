"""Average NYC 2m_T error vs lead, across 32 anchors.

Goal: visualize the small ~2.5% RMSE improvement (which is invisible on a
single-anchor trace). Average |baseline-truth| and |full-truth| across
all anchors at each lead step. Also show mean signed error to expose drift.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/results/2026-06-05-city-traces")
JSON_PATH = DATA_DIR / "v22_K22_nyc_multi_anchor.json"
OUT_DIR = DATA_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    d = json.load(open(JSON_PATH))
    K = d["target_steps"]
    n = d["n_anchors"]
    truth = np.asarray(d["truth_per_anchor"])         # (n, K)
    baseline = np.asarray(d["baseline_per_anchor"])
    full = np.asarray(d["full_per_anchor"])
    lead_hours = (np.arange(K) + 1) * 6.0
    lead_days = lead_hours / 24.0

    err_b = baseline - truth   # (n, K)
    err_f = full - truth

    # Aggregate stats
    mean_abs_b = np.mean(np.abs(err_b), axis=0)
    mean_abs_f = np.mean(np.abs(err_f), axis=0)
    mean_sig_b = np.mean(err_b, axis=0)
    mean_sig_f = np.mean(err_f, axis=0)
    sem_abs_b = np.std(np.abs(err_b), axis=0) / np.sqrt(n)
    sem_abs_f = np.std(np.abs(err_f), axis=0) / np.sqrt(n)
    rmse_b = np.sqrt(np.mean(err_b**2, axis=0))
    rmse_f = np.sqrt(np.mean(err_f**2, axis=0))

    # Per-anchor reduction = |err_b| - |err_f|. Positive = Mamba better.
    per_anchor_diff = np.abs(err_b) - np.abs(err_f)   # (n, K)
    mean_diff = np.mean(per_anchor_diff, axis=0)
    sem_diff = np.std(per_anchor_diff, axis=0) / np.sqrt(n)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) MAE (lat-irrelevant, point-eval): baseline vs Mamba
    ax = axes[0, 0]
    ax.plot(lead_days, mean_abs_b, "C0-", lw=2, label="baseline (GraphCast)")
    ax.fill_between(lead_days, mean_abs_b - sem_abs_b, mean_abs_b + sem_abs_b,
                    color="C0", alpha=0.2)
    ax.plot(lead_days, mean_abs_f, "C3-", lw=2, label="Mamba v22 K=22")
    ax.fill_between(lead_days, mean_abs_f - sem_abs_f, mean_abs_f + sem_abs_f,
                    color="C3", alpha=0.2)
    ax.set_xlabel("lead time (days)")
    ax.set_ylabel("MAE 2m_T at NYC (K)")
    ax.set_title(f"(a) Mean |error| over {n} anchors (±SEM)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Per-anchor reduction = |baseline_err| - |Mamba_err|
    ax = axes[0, 1]
    ax.plot(lead_days, mean_diff, "C2-", lw=2, label="|baseline err| - |Mamba err|")
    ax.fill_between(lead_days, mean_diff - sem_diff, mean_diff + sem_diff,
                    color="C2", alpha=0.25)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("lead time (days)")
    ax.set_ylabel("error reduction (K)")
    ax.set_title("(b) Mamba advantage = how much smaller |Mamba err| is\n"
                 "(positive = Mamba wins; ±SEM)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) Signed error / bias drift
    ax = axes[1, 0]
    ax.plot(lead_days, mean_sig_b, "C0-", lw=2, label="baseline bias")
    ax.plot(lead_days, mean_sig_f, "C3-", lw=2, label="Mamba bias")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("lead time (days)")
    ax.set_ylabel("mean signed error 2m_T (K)")
    ax.set_title(f"(c) Mean signed error over {n} anchors (bias drift)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (d) RMSE
    ax = axes[1, 1]
    ax.plot(lead_days, rmse_b, "C0-", lw=2, label="baseline")
    ax.plot(lead_days, rmse_f, "C3-", lw=2, label="Mamba")
    rel = (rmse_f - rmse_b) / rmse_b * 100
    ax2 = ax.twinx()
    ax2.plot(lead_days, rel, "k--", lw=1.2, alpha=0.7,
             label="rel ΔRMSE (%, right axis)")
    ax2.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax2.set_ylabel("rel ΔRMSE = (Mamba - base) / base (%)")
    ax.set_xlabel("lead time (days)")
    ax.set_ylabel("RMSE 2m_T at NYC (K)")
    ax.set_title("(d) NYC point-RMSE")
    ax.legend(loc="upper left")
    ax2.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle(f"NYC 2m_temperature: v22 K=22 vs baseline, averaged over "
                 f"{n} anchors (val_year=2022)\n"
                 f"single-anchor improvement is buried in synoptic variability; "
                 f"averaging exposes it",
                 fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "nyc_multi_anchor_v22_K22.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # Quick text summary
    print("\n=== summary ===")
    for k in [3, 7, 15, 23, 31, 39]:
        d_hr = (k + 1) * 6
        d_dy = d_hr / 24
        print(f"  lead {d_dy:.1f}d ({d_hr:3d}h)"
              f"  MAE: base={mean_abs_b[k]:.3f}  mamba={mean_abs_f[k]:.3f}"
              f"  diff={mean_diff[k]:+.3f}±{sem_diff[k]:.3f}K"
              f"  ΔRMSE={ (rmse_f[k]-rmse_b[k])/rmse_b[k]*100:+.2f}%")


if __name__ == "__main__":
    main()
