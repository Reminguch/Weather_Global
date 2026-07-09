"""Plot total/weighted-sum improvement and normalised RMSE across all variables.

Two figures:
  1) per-K mean improvement % vs lead (mean across all 11 target vars, equal weight)
     with min-max envelope across vars
  2) per-K normalised total RMSE vs lead. RMSE per var is divided by the baseline's
     own RMSE at the SAME lead (so each var contributes ~1 in baseline → curves are
     comparable). Sum across vars, then divide by n_vars to get a unit-free score.
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
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 12, 16, 20])
    p.add_argument("--prefix", default="eval_res2_2way")
    p.add_argument("--title-suffix", default="res=2 gc_mamba vs GCv1 K=3 baseline")
    return p.parse_args()


def find_json(in_dir, k, prefix):
    if k == 1:
        p = Path(in_dir) / f"{prefix}.json"
        if p.exists(): return p
    gs = sorted(glob.glob(str(Path(in_dir) / f"{prefix}_K{k}_step*.json")))
    # Exclude warmup-sweep files (which have _W{N} in name) from K-sweep glob
    gs = [g for g in gs if "_W" not in Path(g).stem.split(f"_K{k}_step")[1]]
    return Path(gs[-1]) if gs else None


def main():
    cfg = parse_args()
    OUT = Path(cfg.out_dir); OUT.mkdir(parents=True, exist_ok=True)

    data = {}
    for k in cfg.ks:
        p = find_json(cfg.in_dir, k, cfg.prefix)
        if p is None:
            print(f"[skip] K={k}"); continue
        data[k] = json.load(p.open())
        print(f"K={k}: {p.name}")
    if not data: print("no data"); return

    cmap = plt.get_cmap("viridis")
    k_color = {k: cmap(i / max(1, len(cfg.ks) - 1)) for i, k in enumerate(cfg.ks)}

    ref = data[sorted(data.keys())[0]]
    K_t = ref["target_steps"]
    lead_days = (np.arange(1, K_t + 1) * 6) / 24.0
    target_vars = ref["target_variables"]

    # ---- Figure 1: per-K mean improvement % across all vars, with envelope ----
    fig, ax = plt.subplots(figsize=(11, 6))
    for k in cfg.ks:
        if k not in data: continue
        d = data[k]
        all_imps = np.stack([
            np.asarray(d["per_variable_per_lead"][v]["improvement_gc_mamba_pct"])
            for v in target_vars
        ])  # shape (n_vars, K_t)
        mean_imp = all_imps.mean(axis=0)
        lo, hi = all_imps.min(axis=0), all_imps.max(axis=0)
        ax.plot(lead_days, mean_imp, "-o", lw=1.8, markersize=4,
                color=k_color[k], label=f"K={k}")
        ax.fill_between(lead_days, lo, hi, color=k_color[k], alpha=0.10)
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(alpha=0.3)
    ax.set_xlabel("lead (days)")
    ax.set_ylabel(f"mean improvement % across {len(target_vars)} variables")
    ax.set_title(
        f"Total improvement vs baseline — mean over {len(target_vars)} variables (equal weight)\n"
        f"{cfg.title_suffix}", fontsize=11)
    ax.legend(loc="best", fontsize=9, ncol=2)
    p1 = OUT / "total_improvement_vs_lead.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p1}")

    # ---- Figure 2: normalised total RMSE vs lead ----
    # For each var v: norm_rmse_v(lead) = rmse_v(lead) / baseline_rmse_v(lead)
    # Total = mean over v of norm_rmse_v(lead). Baseline ≡ 1.0.
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(1.0, color="k", lw=0.8, label="baseline (= 1.0)")
    for k in cfg.ks:
        if k not in data: continue
        d = data[k]
        norm_rmses = []
        for v in target_vars:
            pv = d["per_variable_per_lead"][v]
            b = np.asarray(pv["rmse_baseline"])
            g = np.asarray(pv["rmse_gc_mamba"])
            norm_rmses.append(g / np.maximum(b, 1e-12))
        norm_rmses = np.stack(norm_rmses)  # (n_vars, K_t)
        total = norm_rmses.mean(axis=0)
        ax.plot(lead_days, total, "-o", lw=1.8, markersize=4,
                color=k_color[k], label=f"K={k}")
    ax.grid(alpha=0.3)
    ax.set_xlabel("lead (days)")
    ax.set_ylabel(f"mean (gc_mamba RMSE / baseline RMSE) across {len(target_vars)} vars")
    ax.set_title(
        f"Total normalised RMSE — gc_mamba relative to baseline (lower = better)\n"
        f"{cfg.title_suffix}", fontsize=11)
    ax.legend(loc="best", fontsize=9, ncol=2)
    p2 = OUT / "total_normalised_rmse_vs_lead.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p2}")

    # ---- Figure 3: lead-mean total improvement vs K (one point per K) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ks_sorted = sorted(data.keys())
    y_means, y_d1, y_d10 = [], [], []
    for k in ks_sorted:
        all_imps = np.stack([
            np.asarray(data[k]["per_variable_per_lead"][v]["improvement_gc_mamba_pct"])
            for v in target_vars
        ])
        y_means.append(float(all_imps.mean()))
        y_d1.append(float(all_imps[:, :4].mean()))      # mean over vars, lead day 1
        y_d10.append(float(all_imps[:, -4:].mean()))    # mean over vars, lead day 10
    ax.plot(ks_sorted, y_d1, "-o", lw=1.6, label="lead day 1 (steps 1-4)")
    ax.plot(ks_sorted, y_means, "-s", lw=1.6, label="lead-mean (all 40 steps)")
    ax.plot(ks_sorted, y_d10, "-^", lw=1.6, label="lead day 10 (steps 37-40)")
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(alpha=0.3)
    ax.set_xlabel("training K (= AR-tail steps in BPTT trunk)")
    ax.set_ylabel(f"mean improvement % across {len(target_vars)} variables")
    ax.set_title(f"Total improvement vs training K — {cfg.title_suffix}", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    p3 = OUT / "total_improvement_vs_trainK.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p3}")


if __name__ == "__main__":
    main()
