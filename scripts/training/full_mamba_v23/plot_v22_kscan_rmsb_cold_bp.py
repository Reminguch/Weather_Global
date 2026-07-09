"""Plot v22 K-scan RMSB figures from cold_bp eval (clean v22-era graphcast.py).

Input JSONs: /home/lm8598/.../results/2026-06-19-v22-rmsb-rerun/v22_K{K}_cold_bp.json
Each has rmsb_baseline, rmsb_full, improvement_pct_rmsb × 40 lead × 11 variables.

Output: per-variable improvement curves, total normalised RMSB vs lead,
        improvement vs train K (3 lead windows), K=22 4-metrics summary.
"""
from __future__ import annotations
import argparse, glob, json, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Only SURFACE variables have RMSB accumulators (per-cell bias for upper-level
# vars is computed per-level and stored separately).
SURFACE = ["2m_temperature","total_precipitation_6hr","mean_sea_level_pressure",
           "10m_u_component_of_wind","10m_v_component_of_wind"]
ALL_VARS = SURFACE  # RMSB only available on surface vars


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="/home/lm8598/Weather_Global_experiments/results/2026-06-19-v22-rmsb-rerun")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ks", nargs="+", type=int, default=[1,2,4,6,8,10,12,14,16,18,20,22])
    cfg = p.parse_args()
    OUT = Path(cfg.out_dir); OUT.mkdir(parents=True, exist_ok=True)

    data = {}
    for k in cfg.ks:
        f = Path(cfg.in_dir) / f"v22_K{k}_cold_bp.json"
        if not f.exists():
            print(f"skip K={k}: {f}"); continue
        data[k] = json.load(f.open())
        print(f"K={k}: {f.name}")
    if not data: return

    cmap = plt.get_cmap("viridis")
    k_color = {k: cmap(i / max(1, len(cfg.ks) - 1)) for i, k in enumerate(cfg.ks)}

    K_t = data[cfg.ks[0]]["target_steps"]
    lead_days = (np.arange(1, K_t + 1) * 6) / 24.0
    lead_hours = np.arange(1, K_t + 1) * 6

    # ---- 1) per-K RMSB improvement vs lead, 5 surface vars overlaid ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    plot_vars = SURFACE
    for vi, var in enumerate(plot_vars):
        ax = axes[vi // 3, vi % 3]
        for k in cfg.ks:
            if k not in data: continue
            pv = data[k]['per_variable_per_step'][var]
            imp = np.asarray(pv['improvement_pct_rmsb'])
            ax.plot(lead_days, imp, "-o", lw=1.2, markersize=2.5,
                    color=k_color[k], label=f"K={k}")
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
        ax.set_title(var, fontsize=10)
        ax.set_xlabel("lead (days)")
        if vi % 3 == 0:
            ax.set_ylabel("RMSB improvement % vs baseline")
        if vi == 0:
            ax.set_visible(True)
        if vi >= len(plot_vars) and vi <= 5:
            ax.legend(fontsize=7, loc="best", ncol=2)
    fig.suptitle("v22 K-scan — per-K RMSB improvement vs lead\n"
                 "cold_bp eval (open loop), v22-era graphcast.py, vs DeepMind GraphCast-small baseline",
                 fontsize=11)
    plt.tight_layout()
    p1 = OUT / "v22_kscan_rmsb_improvement_vs_lead.png"
    plt.savefig(p1, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p1}")

    # ---- 2) total normalised RMSB vs lead, one curve per K ----
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(1.0, color="k", lw=0.8, label="baseline (= 1.0)")
    for k in cfg.ks:
        if k not in data: continue
        d = data[k]
        ratios = []
        for v in ALL_VARS:
            pv = d['per_variable_per_step'][v]
            b = np.asarray(pv['rmsb_baseline'])
            f = np.asarray(pv['rmsb_full'])
            ratios.append(f / np.maximum(b, 1e-12))
        ratios = np.stack(ratios)
        total = ratios.mean(axis=0)
        ax.plot(lead_days, total, "-o", lw=1.5, markersize=3,
                color=k_color[k], label=f"K={k}")
    ax.grid(alpha=0.3)
    ax.set_xlabel("lead (days)")
    ax.set_ylabel(f"mean (v22 RMSB / baseline RMSB) across {len(ALL_VARS)} vars")
    ax.set_title("v22 K-scan — total normalised RMSB vs lead (lower = better)\n"
                 "cold_bp eval, vs DeepMind GraphCast-small baseline", fontsize=11)
    ax.legend(loc="best", fontsize=9, ncol=2)
    plt.tight_layout()
    p2 = OUT / "v22_kscan_total_normalised_rmsb_vs_lead.png"
    plt.savefig(p2, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p2}")

    # ---- 3) total RMSB improvement vs trainK (3 lead windows) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ks_sorted = sorted(data.keys())
    means, d1s, d10s = [], [], []
    for k in ks_sorted:
        d = data[k]
        all_imps = np.stack([
            np.asarray(d['per_variable_per_step'][v]['improvement_pct_rmsb'])
            for v in ALL_VARS
        ])
        means.append(float(all_imps.mean()))
        d1s.append(float(all_imps[:, :4].mean()))
        d10s.append(float(all_imps[:, -4:].mean()))
    ax.plot(ks_sorted, d1s, "-o", lw=1.6, label="lead day 1")
    ax.plot(ks_sorted, means, "-s", lw=1.6, label="lead-mean (all 40 steps)")
    ax.plot(ks_sorted, d10s, "-^", lw=1.6, label="lead day 10")
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(alpha=0.3)
    ax.set_xlabel("training K (AR-tail steps)")
    ax.set_ylabel(f"mean RMSB improvement % across {len(ALL_VARS)} vars")
    ax.set_title("v22 K-scan — total RMSB improvement vs training K (cold_bp eval)", fontsize=12)
    ax.legend(loc="best")
    plt.tight_layout()
    p3 = OUT / "v22_kscan_total_rmsb_improvement_vs_trainK.png"
    plt.savefig(p3, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p3}")

    # ---- 4) K=22 4-metrics panel (May 23 style but for RMSB) ----
    if 22 in data:
        d = data[22]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # (0,0) Total weighted RMSB improvement (normalize per var by its baseline @ d=1)
        ax = axes[0,0]
        rmsb_b = np.array([d['per_variable_per_step'][v]['rmsb_baseline'] for v in ALL_VARS])
        rmsb_f = np.array([d['per_variable_per_step'][v]['rmsb_full'] for v in ALL_VARS])
        scale = rmsb_b[:, 0:1]
        norm_b = (rmsb_b / np.maximum(scale, 1e-12)) ** 2
        norm_f = (rmsb_f / np.maximum(scale, 1e-12)) ** 2
        total_imp = 100 * (1 - norm_f.mean(axis=0) / np.maximum(norm_b.mean(axis=0), 1e-12))
        ax.plot(lead_hours, total_imp, "-", color="C2", lw=1.6, marker="o", markersize=4)
        ax.fill_between(lead_hours, 0, total_imp, where=(total_imp > 0), color="C2", alpha=0.20)
        ax.fill_between(lead_hours, 0, total_imp, where=(total_imp < 0), color="C3", alpha=0.20)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
        ax.set_xticks(np.arange(0, 250, 24))
        ax.set_xlabel("lead (h)"); ax.set_ylabel("improvement %")
        ax.set_title("Metric 1: Total normalised RMSB improvement")

        # (0,1) Per-variable RMSB @ K=4d (lead 16 = step 16)
        ax = axes[0,1]
        lead_96h = 15
        labels = ["2m_T","precip","MSL","10m_u","10m_v"]
        b_rel = np.ones(len(ALL_VARS))
        f_rel = []
        for v in ALL_VARS:
            pv = d['per_variable_per_step'][v]
            b = pv['rmsb_baseline'][lead_96h]; f = pv['rmsb_full'][lead_96h]
            f_rel.append(f / max(b, 1e-12))
        x = np.arange(len(labels)); w = 0.4
        ax.bar(x - w/2, b_rel, w, color="gray", alpha=0.8, label="GC baseline (norm)")
        ax.bar(x + w/2, f_rel, w, color="C2", alpha=0.8, label="v22 K=22 (rel.)")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_ylim([0.5, 1.3])
        ax.grid(alpha=0.3, axis='y')
        ax.legend(fontsize=9)
        ax.set_ylabel("RMSB / baseline_RMSB @ 96h")
        ax.set_title("Metric 2: Per-variable RMSB @ K=4d (lower better)")

        # (1,0) Channels improved (RMSB)
        ax = axes[1,0]
        n_channels = len(d['per_channel_per_step'])
        improved = []
        for k_lead in range(K_t):
            cnt = 0
            for chan, pv in d['per_channel_per_step'].items():
                if 'rmsb_full' in pv and 'rmsb_baseline' in pv:
                    if pv['rmsb_full'][k_lead] < pv['rmsb_baseline'][k_lead]:
                        cnt += 1
            improved.append(cnt)
        ax.plot(lead_hours, improved, "-d", lw=1.4, markersize=4, color="C0")
        ax.axhline(n_channels, color="k", lw=0.5, ls=":", label=f"max ({n_channels}/{n_channels})")
        ax.set_xticks(np.arange(0, 250, 24))
        ax.set_xlabel("lead (h)"); ax.set_ylabel(f"# channels improved (of {n_channels})")
        ax.set_title("Metric 3: # channels with RMSB < baseline_RMSB")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        # (1,1) Per-variable RMSB improvement @ K=4d
        ax = axes[1,1]
        imps = []
        for v in ALL_VARS:
            pv = d['per_variable_per_step'][v]
            b = pv['rmsb_baseline'][lead_96h]; f = pv['rmsb_full'][lead_96h]
            imps.append(100 * (1 - f / max(b, 1e-12)))
        ax.bar(x, imps, color=["C2" if i >= 0 else "C3" for i in imps])
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylabel("improvement %")
        ax.set_title("Metric 4: Per-variable RMSB improvement @ K=4d")

        fig.suptitle("v22 K=22 — RMSB 4 metrics (cold_bp eval, lead 6-240h)", fontsize=12)
        plt.tight_layout()
        p4 = OUT / "v22_K22_4metrics_rmsb.png"
        plt.savefig(p4, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"saved {p4}")

    # ---- 5) CSV: per-K × per-day RMSB improvement ----
    days = list(range(1, 11))
    csv_path = OUT / "v22_rmsb_improvement_trainK_vs_lead_day.csv"
    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["variable","train_K"] + [f"d{d}" for d in days])
        for var in ALL_VARS:
            for k in ks_sorted:
                imp = np.asarray(data[k]['per_variable_per_step'][var]['improvement_pct_rmsb'])
                row = [var, k] + [round(float(imp[4*(d-1):4*d].mean()), 2) for d in days]
                w.writerow(row)
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
