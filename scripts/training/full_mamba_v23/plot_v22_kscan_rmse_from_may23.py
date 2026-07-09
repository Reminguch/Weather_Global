"""Plot v22 K-scan RMSE figures from May 23 JSONs.

Inputs (canonical, locked):
  /home/lm8598/Weather_Global_experiments/results/2026-05-23-v22/eval_jsons/v22_K{K}_K40.json
  Each has 40-lead × 11-variable rmse_baseline, rmse_full, improvement_pct_rmse.

Outputs (this folder):
  per-K improvement_vs_lead, per-K 4metrics, K-sweep K=22 multi-var,
  total improvement vs lead/trainK across K.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir",
        default="/home/lm8598/Weather_Global_experiments/results/2026-05-23-v22/eval_jsons")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ks", nargs="+", type=int,
                   default=[1,2,4,6,8,10,12,14,16,18,20,22])
    return p.parse_args()


SURFACE = ["2m_temperature","total_precipitation_6hr","mean_sea_level_pressure",
           "10m_u_component_of_wind","10m_v_component_of_wind"]
ALL_VARS = SURFACE + ["temperature","specific_humidity","geopotential",
                       "u_component_of_wind","v_component_of_wind","vertical_velocity"]


def main():
    cfg = parse_args()
    OUT = Path(cfg.out_dir); OUT.mkdir(parents=True, exist_ok=True)

    data = {}
    for k in cfg.ks:
        p = Path(cfg.in_dir) / f"v22_K{k}_K40.json"
        if not p.exists():
            print(f"[skip] K={k}: missing {p}"); continue
        data[k] = json.load(p.open())
        print(f"K={k}: {p.name}")
    if not data: return

    cmap = plt.get_cmap("viridis")
    k_color = {k: cmap(i / max(1, len(cfg.ks) - 1)) for i, k in enumerate(cfg.ks)}

    K_t = data[cfg.ks[0]]["target_steps"]
    lead_days = (np.arange(1, K_t + 1) * 6) / 24.0
    lead_hours = (np.arange(1, K_t + 1) * 6)

    # ---------- 1) Per-K K=22 4-metrics plot (matches 2026-05-23 style) ----------
    if 22 in data:
        d = data[22]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # Metric 1: Total weighted MSE improvement (paper formula)
        ax = axes[0, 0]
        # Compute weighted MSE improvement: imp = 1 - sum(rmse_full^2 * w) / sum(rmse_base^2 * w)
        # Equal weights for simplicity (same as paper's spirit, no per-var weighting available here)
        rmse_b = np.array([d['per_variable_per_step'][v]['rmse_baseline'] for v in ALL_VARS])  # (n_vars, K_t)
        rmse_f = np.array([d['per_variable_per_step'][v]['rmse_full'] for v in ALL_VARS])
        # normalise each var by its own baseline at d=1 to remove unit-scale differences
        scale = rmse_b[:, 0:1]
        norm_b = (rmse_b / np.maximum(scale, 1e-12)) ** 2
        norm_f = (rmse_f / np.maximum(scale, 1e-12)) ** 2
        total_imp = 100 * (1 - norm_f.mean(axis=0) / np.maximum(norm_b.mean(axis=0), 1e-12))
        ax.plot(lead_hours, total_imp, "-", color="C2", lw=1.6, marker="o", markersize=4)
        ax.fill_between(lead_hours, 0, total_imp, where=(total_imp > 0), color="C2", alpha=0.20)
        ax.fill_between(lead_hours, 0, total_imp, where=(total_imp < 0), color="C3", alpha=0.20)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
        ax.set_xticks(np.arange(0, 250, 24))
        ax.set_xlabel("lead time (h)"); ax.set_ylabel("improvement %")
        ax.set_title("Metric 1: Total weighted MSE improvement (paper formula)")

        # Metric 2: Per-variable RMSE @ K=4d (lead 16 = step 16 of 40)
        ax = axes[0, 1]
        lead_96h = 15  # 0-indexed lead step 15 = 96h
        labels = ["10m_u","10m_v","2m_T","geopot","MSL","q","temp","precip","u","v","ω"]
        b_rel = np.ones(len(ALL_VARS))
        f_rel = []
        for v in ALL_VARS:
            pv = d['per_variable_per_step'][v]
            rb = pv['rmse_baseline'][lead_96h]
            rf = pv['rmse_full'][lead_96h]
            f_rel.append(rf / max(rb, 1e-12))
        x = np.arange(len(labels)); w = 0.4
        ax.bar(x - w/2, b_rel, w, color="gray", alpha=0.8, label="GC baseline (norm)")
        ax.bar(x + w/2, f_rel, w, color="C2", alpha=0.8, label="v22 K=22 (rel.)")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_ylim([0.6, 1.05])
        ax.grid(alpha=0.3, axis='y')
        ax.legend(fontsize=9)
        ax.set_ylabel("RMSE / baseline_RMSE @ 96h")
        ax.set_title("Metric 2: Per-variable RMSE @ K=4d (lower better)")

        # Metric 3: Channels improved
        ax = axes[1, 0]
        n_channels = len(d['per_channel_per_step'])
        improved = []
        for k_lead in range(K_t):
            cnt = 0
            for chan, pv in d['per_channel_per_step'].items():
                if pv['rmse_full'][k_lead] < pv['rmse_baseline'][k_lead]:
                    cnt += 1
            improved.append(cnt)
        ax.plot(lead_hours, improved, "-d", lw=1.4, markersize=4, color="C0")
        ax.axhline(n_channels, color="k", lw=0.5, ls=":", label=f"max ({n_channels}/{n_channels})")
        ax.set_xticks(np.arange(0, 250, 24))
        ax.set_xlabel("lead time (h)"); ax.set_ylabel(f"# channels improved (of {n_channels})")
        ax.set_title("Metric 3: Channels improved")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        # Metric 4: Per-variable improvement @ K=4d
        ax = axes[1, 1]
        imps = []
        for v in ALL_VARS:
            pv = d['per_variable_per_step'][v]
            rb = pv['rmse_baseline'][lead_96h]; rf = pv['rmse_full'][lead_96h]
            imps.append(100 * (1 - rf / max(rb, 1e-12)))
        ax.bar(x, imps, color=["C2" if i >= 0 else "C3" for i in imps])
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylabel("improvement %")
        ax.set_title("Metric 4: Per-variable improvement @ K=4d")

        fig.suptitle("v22 K=22 — RMSE eval (May 23 JSONs), 4 metrics, lead 6-240h", fontsize=12)
        plt.tight_layout()
        p_out = OUT / "v22_K22_4metrics_rmse.png"
        plt.savefig(p_out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"saved {p_out}")

    # ---------- 2) Per-K improvement vs lead, all K overlay ----------
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    plot_vars = ["2m_temperature","total_precipitation_6hr","mean_sea_level_pressure",
                 "10m_u_component_of_wind","specific_humidity","temperature"]
    for vi, var in enumerate(plot_vars):
        ax = axes[vi // 3, vi % 3]
        for k in cfg.ks:
            if k not in data: continue
            pv = data[k]['per_variable_per_step'][var]
            imp = np.asarray(pv['improvement_pct_rmse'])
            ax.plot(lead_days, imp, "-o", lw=1.2, markersize=2.5,
                    color=k_color[k], label=f"K={k}")
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
        ax.set_title(var, fontsize=10)
        ax.set_xlabel("lead (days)")
        if vi % 3 == 0:
            ax.set_ylabel("RMSE improvement % vs baseline")
        if vi == 0:
            ax.legend(fontsize=7, loc="best", ncol=2)
    fig.suptitle("v22 K-scan — per-K RMSE improvement vs lead (May 23 JSONs)\n"
                 "vs DeepMind GraphCast-small baseline (res=1, mp=16)", fontsize=12)
    plt.tight_layout()
    p_out = OUT / "v22_kscan_improvement_vs_lead.png"
    plt.savefig(p_out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p_out}")

    # ---------- 3) Total normalised RMSE vs lead, one curve per K ----------
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(1.0, color="k", lw=0.8, label="baseline (= 1.0)")
    for k in cfg.ks:
        if k not in data: continue
        d = data[k]
        ratios = []
        for v in ALL_VARS:
            pv = d['per_variable_per_step'][v]
            b = np.asarray(pv['rmse_baseline'])
            f = np.asarray(pv['rmse_full'])
            ratios.append(f / np.maximum(b, 1e-12))
        ratios = np.stack(ratios)  # (n_vars, K_t)
        total = ratios.mean(axis=0)
        ax.plot(lead_days, total, "-o", lw=1.6, markersize=3,
                color=k_color[k], label=f"K={k}")
    ax.grid(alpha=0.3)
    ax.set_xlabel("lead (days)"); ax.set_ylabel(f"mean (v22 RMSE / baseline RMSE) across {len(ALL_VARS)} vars")
    ax.set_title("v22 K-scan — total normalised RMSE vs lead (lower = better)\n"
                 "May 23 JSONs, vs DeepMind GraphCast-small (res=1)", fontsize=11)
    ax.legend(loc="best", fontsize=9, ncol=2)
    plt.tight_layout()
    p_out = OUT / "v22_total_normalised_rmse_vs_lead.png"
    plt.savefig(p_out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p_out}")

    # ---------- 4) Total improvement vs training K (3 lead windows) ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    ks_sorted = sorted(data.keys())
    means, d1s, d10s = [], [], []
    for k in ks_sorted:
        d = data[k]
        all_imps = np.stack([
            np.asarray(d['per_variable_per_step'][v]['improvement_pct_rmse'])
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
    ax.set_ylabel(f"mean RMSE improvement % across {len(ALL_VARS)} vars")
    ax.set_title("v22 K-scan — total improvement vs training K (May 23 JSONs)", fontsize=12)
    ax.legend(loc="best")
    plt.tight_layout()
    p_out = OUT / "v22_total_improvement_vs_trainK.png"
    plt.savefig(p_out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p_out}")

    # ---------- 5) Per-K × per-day improvement table (CSV) ----------
    days = list(range(1, 11))
    import csv
    csv_path = OUT / "v22_improvement_trainK_vs_lead_day.csv"
    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["variable","train_K"] + [f"d{d}" for d in days])
        for var in ALL_VARS:
            for k in ks_sorted:
                imp = np.asarray(data[k]['per_variable_per_step'][var]['improvement_pct_rmse'])
                row = [var, k] + [round(float(imp[4*(d-1):4*d].mean()), 2) for d in days]
                w.writerow(row)
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
