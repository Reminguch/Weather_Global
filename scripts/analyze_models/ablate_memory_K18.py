"""Aggregate + plot K=18 step20000 memory ablation results.

Reads JSONs produced by eval_v22_ablate.py and produces a 4-panel figure:
  - RMSE improvement_pct by lead (per mode)
  - residual cosine by lead (per mode)
  - residual gain (|r|/|err_b|) by lead (per mode)
  - state norm (mean ssm + conv) by lead (per mode)

Modes: normal_W0 / reset_ssm_W0 / reset_all_W0 / normal_W24 / stale_W24
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-26-K18-mem-ablate")

TAGS = ["normal_W0", "reset_ssm_W0", "reset_all_W0", "normal_W24", "stale_W24"]
COLORS = {
    "normal_W0":      "C0",
    "reset_ssm_W0":   "C1",
    "reset_all_W0":   "C3",
    "normal_W24":     "C2",
    "stale_W24":      "C4",
}
LABELS = {
    "normal_W0":      "normal (W=0)",
    "reset_ssm_W0":   "reset_ssm/step (W=0)",
    "reset_all_W0":   "reset_all/step (W=0)",
    "normal_W24":     "normal (W=24)",
    "stale_W24":      "stale_W24",
}

def load_all():
    out = {}
    for tag in TAGS:
        p = ROOT / f"K18_step20000_ablate_{tag}.json"
        if not p.exists():
            print(f"[MISSING] {p.name}")
            continue
        with p.open() as f:
            out[tag] = json.load(f)
        print(f"[LOADED] {tag}: mode={out[tag]['ablation_mode']}  W={out[tag]['warmup_steps']}")
    return out

def plot_var(data, var):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for tag in TAGS:
        if tag not in data: continue
        d = data[tag]
        v = d["per_variable_per_step"].get(var)
        rd = d["residual_diagnostics_per_variable"].get(var)
        if v is None or rd is None: continue
        K = d["target_steps"]
        leads = np.arange(1, K + 1) * 6  # hours

        imp  = v["improvement_pct_rmse"]
        cos  = rd["residual_cosine_by_lead"]
        gain = rd["residual_gain_by_lead"]

        axes[0, 0].plot(leads, imp, color=COLORS[tag], label=LABELS[tag], lw=2)
        axes[0, 1].plot(leads, cos, color=COLORS[tag], label=LABELS[tag], lw=2)
        axes[1, 0].plot(leads, gain, color=COLORS[tag], label=LABELS[tag], lw=2)

        # State norm: sum SSM, sum conv (separately)
        sn = d.get("state_norm_per_layer", {})
        ssm_total = np.zeros(K)
        conv_total = np.zeros(K)
        for k_layer, arr in sn.items():
            if "ssm_state" in k_layer:
                ssm_total += np.asarray(arr)
            elif "conv_cache" in k_layer:
                conv_total += np.asarray(arr)
        # Plot SSM solid, conv dotted
        axes[1, 1].plot(leads, ssm_total, color=COLORS[tag], label=LABELS[tag] + " (ssm)", lw=2)
        axes[1, 1].plot(leads, conv_total, color=COLORS[tag], lw=1, ls=":", alpha=0.6)

    axes[0, 0].set_title(f"{var} — improvement_pct_rmse")
    axes[0, 0].set_xlabel("lead (h)"); axes[0, 0].set_ylabel("% improvement")
    axes[0, 0].axhline(0, color="gray", lw=0.5)
    axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title(f"{var} — residual cosine vs (truth − baseline)")
    axes[0, 1].set_xlabel("lead (h)"); axes[0, 1].set_ylabel("cos(r, -err_b)")
    axes[0, 1].axhline(0, color="gray", lw=0.5)
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_title(f"{var} — residual gain ||r|| / ||err_b||")
    axes[1, 0].set_xlabel("lead (h)"); axes[1, 0].set_ylabel("gain")
    axes[1, 0].axhline(1, color="gray", lw=0.5, ls=":")
    axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title("Mamba state L2 norm by lead (solid=SSM total, dotted=conv total)")
    axes[1, 1].set_xlabel("lead (h)"); axes[1, 1].set_ylabel("|h|_2")
    axes[1, 1].grid(True, alpha=0.3); axes[1, 1].legend(fontsize=7)
    axes[1, 1].set_yscale("log")

    plt.tight_layout()
    out_png = ROOT / f"ablate_K18_{var}.png"
    plt.savefig(out_png, dpi=130)
    plt.close()
    print(f"  → {out_png}")

def print_summary_table(data):
    print()
    print("=" * 100)
    print("K=18 step20000 — Memory ablation summary (improvement_pct_rmse by lead)")
    print("=" * 100)
    for var in ["2m_temperature", "10m_u_component_of_wind"]:
        print(f"\n{var}")
        print(f"  {'mode':<24s} {'L1':>8s} {'L5':>8s} {'L10':>8s} {'L20':>8s} {'L30':>8s} {'L40':>8s}")
        for tag in TAGS:
            if tag not in data: continue
            v = data[tag]["per_variable_per_step"].get(var)
            rd = data[tag]["residual_diagnostics_per_variable"].get(var)
            if v is None: continue
            imps = v["improvement_pct_rmse"]
            cos  = rd["residual_cosine_by_lead"]
            line = f"  {LABELS[tag]:<24s}"
            for ki in [0, 4, 9, 19, 29, 39]:
                if ki < len(imps):
                    line += f" {imps[ki]:>7.2f}%"
            print(line)
            # second row: cos
            line = f"  {'  ↳ cos':24s}"
            for ki in [0, 4, 9, 19, 29, 39]:
                if ki < len(cos):
                    line += f" {cos[ki]:>+8.3f}"
            print(line)

if __name__ == "__main__":
    data = load_all()
    if not data:
        print("No JSONs found.")
        raise SystemExit
    print_summary_table(data)
    for var in ["2m_temperature", "10m_u_component_of_wind",
                "geopotential", "temperature"]:
        plot_var(data, var)
    print("done")
