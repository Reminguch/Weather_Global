"""Plot 3-way K=1 deployment eval:
   - Per-city panels: 2m_T + total_precipitation_6hr time series
     (4 lines: truth, baseline, gc_mamba, res_mamba)
   - Per-variable improvement bar chart vs lead time
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-json", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main():
    cfg = parse_args()
    d = json.load(open(cfg.in_json))
    cities = list(d["cities"].keys())
    target_vars = d["target_variables"]
    n_anch = d["n_anchors"]
    K = d["target_steps"]
    lead_hours = np.arange(1, K + 1) * 6   # 6, 12, ..., 240 (10 days)
    lead_days = lead_hours / 24.0

    OUT = Path(cfg.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    # === Plot 1: per-city per-variable mean rollout ===
    plot_vars = [v for v in ["2m_temperature", "total_precipitation_6hr"] if v in target_vars]
    n_cities = len(cities)
    fig, axes = plt.subplots(n_cities, len(plot_vars), figsize=(8*len(plot_vars), 2.4*n_cities),
                             squeeze=False)
    color = {"truth":"k", "baseline":"C0", "gc_mamba":"C1", "res_mamba":"C2"}
    style = {"truth":"-", "baseline":"--", "gc_mamba":"-", "res_mamba":"-"}
    lw = {"truth":2.0, "baseline":1.4, "gc_mamba":1.4, "res_mamba":1.4}

    for ci, city in enumerate(cities):
        for vi, var in enumerate(plot_vars):
            ax = axes[ci, vi]
            for model in ["truth", "baseline", "gc_mamba", "res_mamba"]:
                arr = np.asarray(d["city_traces"][model][city][var])
                if arr.size == 0: continue
                # arr shape (n_anch * K,). Reshape to (n_anch, K), then mean.
                arr = arr.reshape(n_anch, K)
                scale = 1000.0 if var == "total_precipitation_6hr" else 1.0  # m → mm
                mean_traj = arr.mean(axis=0) * scale
                std_traj = arr.std(axis=0) * scale
                ax.plot(lead_days, mean_traj, color=color[model], ls=style[model],
                        lw=lw[model], label=model)
                if model != "truth":
                    ax.fill_between(lead_days, mean_traj - std_traj, mean_traj + std_traj,
                                    color=color[model], alpha=0.1)
            # RMSE annotations (lead-mean RMSE)
            pv = d["per_variable_per_lead"].get(var)
            if pv is not None:
                rmse_b = np.mean(pv["rmse_baseline"])
                rmse_g = np.mean(pv["rmse_gc_mamba"])
                rmse_r = np.mean(pv["rmse_res_mamba"])
                ax.set_title(f"{city} — {var}\nlead-mean RMSE: base={rmse_b:.3f}  "
                             f"gcm={rmse_g:.3f} ({100*(1-rmse_g/rmse_b):+.1f}%)  "
                             f"rm={rmse_r:.3f} ({100*(1-rmse_r/rmse_b):+.1f}%)", fontsize=8)
            else:
                ax.set_title(f"{city} — {var}", fontsize=9)
            ax.set_xlabel("lead (days)"); ax.set_ylabel(var)
            ax.grid(alpha=0.3)
            if ci == 0 and vi == 0:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"3-way K=1 eval: warm-up 24 truth + 40-step closed-loop self-rollout\n"
                 f"({n_anch} anchors, val_year=2022, res=2 GCv1 K=3 baseline)", fontsize=11)
    plt.tight_layout()
    p1 = OUT / "city_traces_3way.png"
    plt.savefig(p1, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p1}")

    # === Plot 2: per-variable improvement vs lead ===
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    surface_vars = ["2m_temperature", "mean_sea_level_pressure",
                    "10m_u_component_of_wind", "10m_v_component_of_wind",
                    "total_precipitation_6hr"]
    atmo_vars = [v for v in target_vars if v not in surface_vars]

    # Panel (0,0): improvement % vs lead for surface vars (gc_mamba)
    for var in surface_vars:
        if var not in d["per_variable_per_lead"]: continue
        imp = np.asarray(d["per_variable_per_lead"][var]["improvement_gc_mamba_pct"])
        axes[0,0].plot(lead_days, imp, "-o", lw=1.3, markersize=4, label=var)
    axes[0,0].axhline(0, color="k", lw=0.5); axes[0,0].grid(alpha=0.3)
    axes[0,0].set_title("gc_mamba improvement % vs baseline (surface vars)")
    axes[0,0].set_xlabel("lead (days)"); axes[0,0].set_ylabel("improvement %")
    axes[0,0].legend(fontsize=8)

    # Panel (0,1): same for res_mamba
    for var in surface_vars:
        if var not in d["per_variable_per_lead"]: continue
        imp = np.asarray(d["per_variable_per_lead"][var]["improvement_res_mamba_pct"])
        axes[0,1].plot(lead_days, imp, "-o", lw=1.3, markersize=4, label=var)
    axes[0,1].axhline(0, color="k", lw=0.5); axes[0,1].grid(alpha=0.3)
    axes[0,1].set_title("res_mamba improvement % vs baseline (surface vars)")
    axes[0,1].set_xlabel("lead (days)"); axes[0,1].set_ylabel("improvement %")
    axes[0,1].legend(fontsize=8)

    # Panel (1,0): bar chart of lead-mean improvement % per variable
    var_labels = []
    gcm_imps = []
    rm_imps = []
    for var in target_vars:
        pv = d["per_variable_per_lead"][var]
        var_labels.append(var.replace("_component_of_wind", "_wind"))
        gcm_imps.append(float(np.mean(pv["improvement_gc_mamba_pct"])))
        rm_imps.append(float(np.mean(pv["improvement_res_mamba_pct"])))
    x = np.arange(len(var_labels))
    w = 0.4
    axes[1,0].bar(x - w/2, gcm_imps, w, label="gc_mamba", color="C1")
    axes[1,0].bar(x + w/2, rm_imps,  w, label="res_mamba", color="C2")
    axes[1,0].axhline(0, color="k", lw=0.5); axes[1,0].grid(alpha=0.3, axis='y')
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(var_labels, rotation=30, ha="right", fontsize=8)
    axes[1,0].set_ylabel("lead-mean improvement % vs baseline")
    axes[1,0].set_title("Per-variable lead-mean improvement")
    axes[1,0].legend()

    # Panel (1,1): cumulative improvement vs lead, averaged over surface vars
    surf_present = [v for v in surface_vars if v in d["per_variable_per_lead"]]
    if surf_present:
        gcm_curves = np.stack([
            d["per_variable_per_lead"][v]["improvement_gc_mamba_pct"] for v in surf_present])
        rm_curves = np.stack([
            d["per_variable_per_lead"][v]["improvement_res_mamba_pct"] for v in surf_present])
        axes[1,1].plot(lead_days, gcm_curves.mean(axis=0), "C1-o", lw=1.6, label="gc_mamba")
        axes[1,1].plot(lead_days, rm_curves.mean(axis=0), "C2-o", lw=1.6, label="res_mamba")
        axes[1,1].axhline(0, color="k", lw=0.5); axes[1,1].grid(alpha=0.3)
        axes[1,1].set_title(f"Mean improvement % across {len(surf_present)} surface vars")
        axes[1,1].set_xlabel("lead (days)"); axes[1,1].set_ylabel("mean improvement %")
        axes[1,1].legend()

    fig.suptitle(f"3-way K=1 deployment improvement vs baseline (GCv1 K=3)\n"
                 f"warm-up=24 truth + 40-step closed-loop, {n_anch} anchors", fontsize=11)
    plt.tight_layout()
    p2 = OUT / "improvement_3way.png"
    plt.savefig(p2, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p2}")

    # Text summary
    print()
    print("=== Per-variable lead-mean improvement (printed) ===")
    print(f"  {'variable':<32}{'rmse_base':>12}{'gc_mamba':>12}{'res_mamba':>12}")
    for var in target_vars:
        pv = d["per_variable_per_lead"][var]
        b = float(np.mean(pv["rmse_baseline"]))
        ig = float(np.mean(pv["improvement_gc_mamba_pct"]))
        ir = float(np.mean(pv["improvement_res_mamba_pct"]))
        print(f"  {var:<32}{b:>12.4f}{ig:>+11.2f}%{ir:>+11.2f}%")


if __name__ == "__main__":
    main()
