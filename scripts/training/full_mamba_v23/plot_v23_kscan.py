#!/usr/bin/env python
"""Generate per-K plots from a single eval JSON.

Produces 3 PNGs into `<plot_dir>/`:
  1. 4metrics_K40.png          — total MSE-imp + per-var RMSE @ K=4 day
  2. per_variable_improvement_K40.png  — 12 per-variable improvement curves vs lead
  3. per_variable_rmse_K40.png — raw RMSE per variable

Modeled after the v22 plots in results/2026-05-23-v22/plots/K{N}/.

Usage:
  python plot_v23_kscan.py <eval_json> <plot_dir>
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Per-variable weights (paper formula)
W_VAR = {
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "2m_temperature": 1.0,
    "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1,
    # 3D vars: GraphCast uses level-pressure-weighting × 1.0
    "geopotential": 1.0,
    "temperature": 1.0,
    "u_component_of_wind": 1.0,
    "v_component_of_wind": 1.0,
    "vertical_velocity": 1.0,
    "specific_humidity": 1.0,
}

# s_v from training stats — used for paper-style normalization in Metric 1.
# Per GraphCast paper § "Loss function" the target is the *residual* (next_step − prev_step),
# normalized by the time-difference stddev (`diffs_stddev_by_level.nc`), NOT the
# climatological stddev. So for the aggregate skill metric to match the training objective,
# we divide MSE_v by s_v² where s_v = diffs_stddev_v (6h tendency stddev).
# Falls back to 1.0 if file missing (raw-sum aggregate, geopotential-dominated).
def _load_diffs_stddev(path="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats/diffs_stddev_by_level.nc"):
    try:
        import xarray as xr
        ds = xr.open_dataset(path)
    except Exception:
        return {}
    out = {}
    for v in W_VAR.keys():
        if v not in ds:
            continue
        arr = ds[v].values
        if arr.ndim == 0:
            out[v] = float(arr)
        else:
            # 3D var: use mean s over pressure levels as representative scale.
            # (Eval JSON's rmse_baseline/full per 3D var is already averaged over levels,
            # so dividing by mean-s is consistent.)
            out[v] = float(arr.mean())
    return out

SIGMA_V = _load_diffs_stddev()


def per_var_mse_imp(ev: dict, var: str, k: int) -> tuple[float, float]:
    """Return (baseline_mse, full_mse) at lead index k (0-indexed)."""
    pv = ev["per_variable_per_step"][var]
    rb = pv["rmse_baseline"][k]
    rf = pv["rmse_full"][k]
    return rb * rb, rf * rf


def plot_4metrics(ev: dict, out_path: Path, K_train: int, version: str = "v23", d_conv: int = 8) -> None:
    """4-panel figure: total MSE-imp vs lead, per-var RMSE @ K=4 day, channels improved, per-var imp@K=4."""
    n_lead = ev["target_steps"]
    target_lead_h = [6 * (i + 1) for i in range(n_lead)]   # 6h, 12h, ..., 240h
    leadlist = np.arange(1, n_lead + 1)
    variables = list(ev["per_variable_per_step"].keys())

    # Metric 1: total weighted MSE-improvement vs lead (PAPER FORMULA, s_v normalized).
    # s_v = diffs_stddev_v (6h tendency stddev), matching GraphCast's training normalizer.
    # Each variable's MSE is divided by s_v² before W_var-weighted sum so that variables
    # with vastly different residual scales contribute comparably to the aggregate.
    imp_total = []
    for k in leadlist:
        tb = tf = 0.0
        for v in variables:
            w = W_VAR.get(v, 1.0)
            sigma = SIGMA_V.get(v, None)
            if sigma is None or sigma <= 0:
                continue  # skip variables without σ
            rb, rf = per_var_mse_imp(ev, v, k - 1)
            tb += w * rb / (sigma * sigma)
            tf += w * rf / (sigma * sigma)
        imp_total.append((1.0 - tf / tb) * 100.0 if tb > 0 else 0.0)

    # Metric 2: per-variable RMSE@K=4 day (lead idx 16 = 4 day, but 6h * 16 = 96h, not exactly 4 day)
    # Use lead idx where target_lead_h == 96 (4 day). With 6h step, idx 15 = 96h.
    lead_4d_idx = 15 if n_lead > 15 else n_lead - 1
    var_rmse_baseline = []
    var_rmse_full = []
    for v in variables:
        pv = ev["per_variable_per_step"][v]
        var_rmse_baseline.append(pv["rmse_baseline"][lead_4d_idx])
        var_rmse_full.append(pv["rmse_full"][lead_4d_idx])

    # Metric 3: channels improved vs lead (per_channel_per_step)
    pc = ev.get("per_channel_per_step", {})
    n_channels = len(pc) if pc else 83
    channels_improved = []
    for k in leadlist:
        if not pc:
            channels_improved.append(0)
            continue
        n_imp = 0
        for ch in pc.values():
            rb = ch["rmse_baseline"][k - 1]
            rf = ch["rmse_full"][k - 1]
            if rf < rb:
                n_imp += 1
        channels_improved.append(n_imp)

    # Metric 4: per-variable improvement% @K=4 day
    var_imp_pct = [
        (1.0 - rf / rb) * 100.0 if rb > 0 else 0.0
        for rb, rf in zip(var_rmse_baseline, var_rmse_full)
    ]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    max_lead_h = target_lead_h[-1]
    fig.suptitle(f"{version} K={K_train} chunk=24 (d_conv={d_conv}) — 4 metrics, lead 6-{max_lead_h}h", fontsize=13)
    # dynamic x-axis ticks based on lead range
    def _dyn_ticks(maxh):
        if maxh <= 72:   step = 12
        elif maxh <= 120: step = 24
        else:             step = 24
        return list(range(0, maxh + 1, step))[1:]
    xticks = _dyn_ticks(max_lead_h)
    axes[0, 0].set_xticks(xticks); axes[0, 0].set_xlim(0, max_lead_h + 6)
    axes[1, 0].set_xticks(xticks); axes[1, 0].set_xlim(0, max_lead_h + 6)

    # M1: total imp curve
    ax = axes[0, 0]
    ax.plot(target_lead_h, imp_total, lw=2, color="C0", marker="o", markersize=3)
    ax.fill_between(target_lead_h, 0, imp_total, where=[x > 0 for x in imp_total],
                    color="C2", alpha=0.30)
    ax.fill_between(target_lead_h, 0, imp_total, where=[x <= 0 for x in imp_total],
                    color="C3", alpha=0.30)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("improvement %")
    ax.set_title("Metric 1: Total weighted MSE improvement (paper formula)")
    ax.grid(alpha=0.3)
    # xticks set dynamically below based on actual lead range

    # M2: per-var rmse RATIO @K=4day (baseline normalized to 1.0; v23 bars show ratio).
    # This is the v22-style display: bars all around 1.0, easy visual comparison.
    ax = axes[0, 1]
    x = np.arange(len(variables))
    w = 0.4
    var_ratio = [rf / rb if rb > 0 else 1.0 for rb, rf in zip(var_rmse_baseline, var_rmse_full)]
    ax.bar(x - w/2, [1.0] * len(variables), w, label="GC baseline (norm)", color="0.7")
    ax.bar(x + w/2, var_ratio, w, label=f"{version} K={K_train} (rel.)", color="C2")
    ax.axhline(1.0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_component_of_wind", "")
                       .replace("_temperature", "_T")
                       .replace("mean_sea_level_pressure", "MSL")
                       .replace("total_precipitation_6hr", "precip")
                       .replace("vertical_velocity", "ω")
                       .replace("specific_humidity", "q")
                       .replace("u_component_of_wind", "u")
                       .replace("v_component_of_wind", "v")
                       for v in variables], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"RMSE / baseline_RMSE @ {target_lead_h[lead_4d_idx]}h")
    ax.set_title(f"Metric 2: Per-variable RMSE @ K={target_lead_h[lead_4d_idx]//24}d (lower better)")
    ax.set_ylim(0, max(1.05, max(var_ratio) * 1.05))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # M3: channels improved vs lead
    ax = axes[1, 0]
    ax.plot(target_lead_h, channels_improved, lw=2, color="C0", marker="d", markersize=4)
    ax.axhline(n_channels, color="k", lw=0.5, ls=":", label=f"max ({n_channels}/{n_channels})")
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel(f"# channels improved (of {n_channels})")
    ax.set_title("Metric 3: Channels improved")
    # xticks set dynamically below based on actual lead range
    ax.set_ylim(0, n_channels + 5)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)

    # M4: per-var improvement %@K=4day
    ax = axes[1, 1]
    colors = ["C2" if x > 0 else "C3" for x in var_imp_pct]
    ax.bar(x, var_imp_pct, color=colors)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_component_of_wind", "")
                       .replace("_temperature", "_T")
                       .replace("mean_sea_level_pressure", "MSL")
                       .replace("total_precipitation_6hr", "precip")
                       .replace("vertical_velocity", "ω")
                       .replace("specific_humidity", "q")
                       .replace("u_component_of_wind", "u")
                       .replace("v_component_of_wind", "v")
                       for v in variables], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("improvement %")
    ax.set_title(f"Metric 4: Per-variable improvement @ K={target_lead_h[lead_4d_idx]//24}d")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_per_variable_improvement(ev: dict, out_path: Path, K_train: int, version: str = "v23", d_conv: int = 8) -> None:
    """3×4 grid of per-variable improvement% vs lead (matches v22 plot style)."""
    n_lead = ev["target_steps"]
    target_lead_h = [6 * (i + 1) for i in range(n_lead)]
    variables = list(ev["per_variable_per_step"].keys())
    # Pad with blanks if <12 variables, else select first 12
    show_vars = variables[:12]
    fig, axes = plt.subplots(3, 4, figsize=(20, 11))
    fig.suptitle(f"{version} K={K_train} chunk=24 (d_conv={d_conv}) — per-variable improvement vs GraphCast baseline (lat-w RMSE)", fontsize=12)
    name_map = {
        "10m_u_component_of_wind": "10m_u", "10m_v_component_of_wind": "10m_v",
        "2m_temperature": "2m_T", "mean_sea_level_pressure": "MSL",
        "total_precipitation_6hr": "precip 6h", "vertical_velocity": "vertical_velocity",
        "geopotential": "geopotential", "temperature": "temperature",
        "u_component_of_wind": "u_wind", "v_component_of_wind": "v_wind",
        "specific_humidity": "q",
    }

    for i, v in enumerate(show_vars):
        ax = axes[i // 4, i % 4]
        pv = ev["per_variable_per_step"][v]
        rb = np.asarray(pv["rmse_baseline"])
        rf = np.asarray(pv["rmse_full"])
        imp = (1.0 - rf / rb) * 100.0
        ax.plot(target_lead_h, imp, lw=2, color="C2")
        ax.fill_between(target_lead_h, 0, imp, where=imp > 0, color="C2", alpha=0.30)
        ax.fill_between(target_lead_h, 0, imp, where=imp <= 0, color="C3", alpha=0.30)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("lead (h)", fontsize=9)
        ax.set_ylabel("improvement % (+ better)", fontsize=9)
        ax.set_title(name_map.get(v, v), fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xticks([50, 100, 150, 200])

    # Hide unused
    for i in range(len(show_vars), 12):
        axes[i // 4, i % 4].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_per_variable_rmse(ev: dict, out_path: Path, K_train: int, version: str = "v23", d_conv: int = 8) -> None:
    """3×4 grid of raw RMSE (baseline vs v23) per variable vs lead."""
    n_lead = ev["target_steps"]
    target_lead_h = [6 * (i + 1) for i in range(n_lead)]
    variables = list(ev["per_variable_per_step"].keys())
    show_vars = variables[:12]
    fig, axes = plt.subplots(3, 4, figsize=(20, 11))
    fig.suptitle(f"{version} K={K_train} chunk=24 (d_conv={d_conv}) — per-variable RMSE vs lead (lat-w)", fontsize=12)
    name_map = {
        "10m_u_component_of_wind": "10m_u", "10m_v_component_of_wind": "10m_v",
        "2m_temperature": "2m_T", "mean_sea_level_pressure": "MSL",
        "total_precipitation_6hr": "precip 6h", "vertical_velocity": "vertical_velocity",
        "geopotential": "geopotential", "temperature": "temperature",
        "u_component_of_wind": "u_wind", "v_component_of_wind": "v_wind",
        "specific_humidity": "q",
    }

    for i, v in enumerate(show_vars):
        ax = axes[i // 4, i % 4]
        pv = ev["per_variable_per_step"][v]
        rb = np.asarray(pv["rmse_baseline"])
        rf = np.asarray(pv["rmse_full"])
        ax.plot(target_lead_h, rb, color="0.5", lw=1.5, label="GC baseline")
        ax.plot(target_lead_h, rf, color="C2", lw=2, label=f"{version} K={K_train}")
        ax.set_xlabel("lead (h)", fontsize=9)
        ax.set_ylabel("RMSE (lat-w)", fontsize=9)
        ax.set_title(name_map.get(v, v), fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xticks([50, 100, 150, 200])

    for i in range(len(show_vars), 12):
        axes[i // 4, i % 4].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    if len(sys.argv) < 3:
        print("Usage: plot_v23_kscan.py <eval_json> <plot_dir> [version] [d_conv]")
        print("  version defaults to inferred from JSON filename prefix (v22/v23 → those, else v23)")
        print("  d_conv defaults to 4 for v22, 8 for v23")
        sys.exit(2)
    json_path = Path(sys.argv[1])
    plot_dir = Path(sys.argv[2])
    plot_dir.mkdir(parents=True, exist_ok=True)

    ev = json.loads(json_path.read_text())
    # Infer K and version from filename like {version}_K{N}_..._K40.json
    name = json_path.stem
    parts = name.split("_")
    version = sys.argv[3] if len(sys.argv) > 3 else (parts[0] if parts[0] in ("v22", "v23") else "v23")
    d_conv = int(sys.argv[4]) if len(sys.argv) > 4 else (4 if version == "v22" else 8)
    K = int(parts[1].lstrip("K")) if len(parts) > 1 and parts[1].startswith("K") else 0

    # Output filename suffix = number of eval lead steps (K20 = 120h, K40 = 240h, etc).
    n_lead = ev["target_steps"]
    suffix = f"K{n_lead}"
    plot_4metrics(ev, plot_dir / f"4metrics_{suffix}.png", K, version, d_conv)
    plot_per_variable_improvement(ev, plot_dir / f"per_variable_improvement_{suffix}.png", K, version, d_conv)
    plot_per_variable_rmse(ev, plot_dir / f"per_variable_rmse_{suffix}.png", K, version, d_conv)
    print(f"saved 3 plots to {plot_dir} (suffix _{suffix})")


if __name__ == "__main__":
    main()
