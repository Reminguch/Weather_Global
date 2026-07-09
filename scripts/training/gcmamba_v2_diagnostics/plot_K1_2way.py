"""Plot 2-way K=1 deployment eval (baseline vs gc_mamba)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--title-prefix", default="gc_mamba K=1 vs baseline")
    return p.parse_args()


def main():
    cfg = parse_args()
    d = json.load(open(cfg.in_json))
    cities = list(d["cities"].keys())
    target_vars = d["target_variables"]
    n_anch = d["n_anchors"]
    K = d["target_steps"]
    lead_days = (np.arange(1, K + 1) * 6) / 24.0

    OUT = Path(cfg.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    # --- 1) City traces: temp + precip per city ---
    plot_vars = [v for v in ["2m_temperature", "total_precipitation_6hr"] if v in target_vars]
    n_cities = len(cities)
    fig, axes = plt.subplots(n_cities, len(plot_vars), figsize=(8*len(plot_vars), 2.4*n_cities), squeeze=False)
    color = {"truth":"k", "baseline":"C0", "gc_mamba":"C1"}
    style = {"truth":"-", "baseline":"--", "gc_mamba":"-"}
    lw = {"truth":2.0, "baseline":1.4, "gc_mamba":1.6}

    for ci, city in enumerate(cities):
        for vi, var in enumerate(plot_vars):
            ax = axes[ci, vi]
            for model in ["truth", "baseline", "gc_mamba"]:
                arr = np.asarray(d["city_traces"][model][city][var])
                if arr.size == 0: continue
                arr = arr.reshape(n_anch, K)
                scale = 1000.0 if var == "total_precipitation_6hr" else 1.0
                mean_traj = arr.mean(axis=0) * scale
                ax.plot(lead_days, mean_traj, color=color[model], ls=style[model],
                        lw=lw[model], label=model if (ci==0 and vi==0) else None)
            pv = d["per_variable_per_lead"].get(var)
            if pv is not None:
                rmse_b = float(np.mean(pv["rmse_baseline"]))
                rmse_g = float(np.mean(pv["rmse_gc_mamba"]))
                imp = 100*(1 - rmse_g/rmse_b)
                ax.set_title(f"{city} — {var}\n"
                             f"lead-mean RMSE: base={rmse_b:.3f}  gcm={rmse_g:.3f} ({imp:+.1f}%)",
                             fontsize=8)
            else:
                ax.set_title(f"{city} — {var}", fontsize=9)
            ax.set_xlabel("lead (days)"); ax.set_ylabel(var)
            ax.grid(alpha=0.3)
            if ci==0 and vi==0:
                ax.legend(fontsize=9, loc="best")
    fig.suptitle(f"{cfg.title_prefix}: warm 24-step truth + 40-step closed-loop self-rollout\n"
                 f"({n_anch} anchors, val_year=2022)", fontsize=11)
    plt.tight_layout()
    p1 = OUT / "city_traces_2way.png"
    plt.savefig(p1, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p1}")

    # --- 2) Per-variable improvement table + bar chart ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    surface_vars = ["2m_temperature","mean_sea_level_pressure",
                    "10m_u_component_of_wind","10m_v_component_of_wind",
                    "total_precipitation_6hr"]

    # (0,0) Surface var improvement vs lead
    for var in surface_vars:
        if var not in d["per_variable_per_lead"]: continue
        imp = np.asarray(d["per_variable_per_lead"][var]["improvement_gc_mamba_pct"])
        axes[0,0].plot(lead_days, imp, "-o", lw=1.3, markersize=4, label=var)
    axes[0,0].axhline(0, color="k", lw=0.5)
    axes[0,0].grid(alpha=0.3)
    axes[0,0].set_title("Surface vars: gc_mamba improvement % vs baseline")
    axes[0,0].set_xlabel("lead (days)"); axes[0,0].set_ylabel("improvement %")
    axes[0,0].legend(fontsize=8, loc="best")

    # (0,1) All vars: bar of lead-mean improvement
    var_labels = []
    imps = []
    for var in target_vars:
        pv = d["per_variable_per_lead"][var]
        var_labels.append(var.replace("_component_of_wind", "_wind"))
        imps.append(float(np.mean(pv["improvement_gc_mamba_pct"])))
    x = np.arange(len(var_labels))
    cs = ['C2' if imp >= 0 else 'C3' for imp in imps]
    axes[0,1].bar(x, imps, color=cs)
    axes[0,1].axhline(0, color="k", lw=0.5)
    axes[0,1].grid(alpha=0.3, axis='y')
    axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(var_labels, rotation=30, ha="right", fontsize=8)
    axes[0,1].set_ylabel("lead-mean improvement %")
    axes[0,1].set_title("Per-variable lead-mean improvement")
    for xi, vi in zip(x, imps):
        axes[0,1].text(xi, vi + (0.05 if vi>=0 else -0.1), f'{vi:+.1f}', ha='center',
                       fontsize=7, va='bottom' if vi>=0 else 'top')

    # (1,0) RMSE absolute curves (baseline vs gcm) for 2m_T
    if "2m_temperature" in d["per_variable_per_lead"]:
        pv = d["per_variable_per_lead"]["2m_temperature"]
        axes[1,0].plot(lead_days, pv["rmse_baseline"], "C0-o", lw=1.4, markersize=4, label="baseline")
        axes[1,0].plot(lead_days, pv["rmse_gc_mamba"], "C1-o", lw=1.4, markersize=4, label="gc_mamba")
        axes[1,0].set_title("2m_temperature RMSE vs lead")
        axes[1,0].set_xlabel("lead (days)"); axes[1,0].set_ylabel("RMSE (K)")
        axes[1,0].grid(alpha=0.3); axes[1,0].legend()

    # (1,1) Mean surface improvement curve
    surf_present = [v for v in surface_vars if v in d["per_variable_per_lead"]]
    if surf_present:
        curves = np.stack([d["per_variable_per_lead"][v]["improvement_gc_mamba_pct"] for v in surf_present])
        axes[1,1].plot(lead_days, curves.mean(axis=0), "C1-o", lw=1.6, label="mean across surface vars")
        axes[1,1].fill_between(lead_days, curves.min(axis=0), curves.max(axis=0),
                               alpha=0.15, color="C1", label="min-max range across vars")
        axes[1,1].axhline(0, color="k", lw=0.5)
        axes[1,1].grid(alpha=0.3); axes[1,1].legend(fontsize=9)
        axes[1,1].set_title(f"Mean surface-var improvement (n={len(surf_present)})")
        axes[1,1].set_xlabel("lead (days)"); axes[1,1].set_ylabel("improvement %")

    fig.suptitle(f"{cfg.title_prefix} — per-variable improvement (closed-loop deployment, "
                 f"{n_anch} anchors × 40-step rollout)", fontsize=11)
    plt.tight_layout()
    p2 = OUT / "improvement_2way.png"
    plt.savefig(p2, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p2}")

    # Text summary
    print()
    print("=== Per-variable lead-mean improvement ===")
    print(f"  {'variable':<32}{'rmse_base':>12}{'rmse_gcm':>12}{'imp %':>10}")
    for var in target_vars:
        pv = d["per_variable_per_lead"][var]
        b = float(np.mean(pv["rmse_baseline"]))
        g = float(np.mean(pv["rmse_gc_mamba"]))
        i = float(np.mean(pv["improvement_gc_mamba_pct"]))
        print(f"  {var:<32}{b:>12.4f}{g:>12.4f}{i:>+9.2f}%")


if __name__ == "__main__":
    main()
