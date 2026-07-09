#!/usr/bin/env python
"""Plot mean traces in two columns: cold (old) vs warm_bp (new).

3 lines each: ERA5 truth (black), baseline (blue dashed), residual+Mamba (red).
Save 83 PNGs to 2026-06-02-warm-bp-mean-traces/plots/.

The cold version has the residual_state bug (loaded ckpt state); the warm_bp
version is clean (zero init + 24-step truth warmup). Direct comparison shows
whether Mamba's drift in cold traces was an eval artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLD_JSON = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/"
                 "results/2026-06-01-mean-traces/v22_K22_means.json")
WARM_JSON = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/"
                 "results/2026-06-02-warm-bp-mean-traces/v22_K22_warm_bp_means.json")
OUT_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/"
               "results/2026-06-02-warm-bp-mean-traces/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNITS = {
    "2m_temperature": "K", "temperature": "K",
    "10m_u_component_of_wind": "m/s", "10m_v_component_of_wind": "m/s",
    "u_component_of_wind": "m/s", "v_component_of_wind": "m/s",
    "mean_sea_level_pressure": "Pa",
    "total_precipitation_6hr": "m",
    "geopotential": "m^2/s^2",
    "specific_humidity": "kg/kg",
    "vertical_velocity": "Pa/s",
}


def unit_for(channel: str) -> str:
    base = channel.split("_level")[0]
    return UNITS.get(base, "")


def main() -> None:
    cold = json.load(open(COLD_JSON))
    warm = json.load(open(WARM_JSON))
    K_cold = cold["target_steps"]
    K_warm = warm["target_steps"]
    leads_h_cold = np.arange(1, K_cold + 1) * 6
    leads_h_warm = np.arange(1, K_warm + 1) * 6

    common_chans = sorted(set(cold["per_channel"]) & set(warm["per_channel"]))
    print(f"plotting {len(common_chans)} channels (intersection of cold & warm)")

    for chan in common_chans:
        u = unit_for(chan)

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)

        # LEFT: COLD eval
        d = cold["per_channel"][chan]
        ax_l.plot(leads_h_cold, d["truth"], "k-", lw=2.0, label="ERA5 truth")
        ax_l.plot(leads_h_cold, d["baseline"], "C0--", lw=1.6, label="baseline (GraphCast)")
        ax_l.plot(leads_h_cold, d["full"], "C3-", lw=1.6, label="residual+Mamba (v22 K=22)")
        ax_l.set_xlabel("lead time (h)")
        ax_l.set_ylabel(f"lat-weighted global mean ({u})" if u else "lat-weighted global mean")
        ax_l.set_title("COLD eval (cold-start, bp-feedback, BUGGY ckpt state)")
        ax_l.grid(alpha=0.3)
        ax_l.legend(loc="best", fontsize=8)

        # RIGHT: WARM_BP eval (clean)
        d = warm["per_channel"][chan]
        ax_r.plot(leads_h_warm, d["truth"], "k-", lw=2.0, label="ERA5 truth")
        ax_r.plot(leads_h_warm, d["baseline"], "C0--", lw=1.6, label="baseline (GraphCast)")
        ax_r.plot(leads_h_warm, d["full"], "C3-", lw=1.6, label="residual+Mamba (v22 K=22)")
        ax_r.set_xlabel("lead time (h)")
        ax_r.set_title("WARM_BP eval (24-step truth warmup, bp-feedback, CLEAN state)")
        ax_r.grid(alpha=0.3)
        ax_r.legend(loc="best", fontsize=8)

        fig.suptitle(f"{chan} — global mean over 32 anchors (2022). v22 K=22 ckpt.",
                     fontsize=11)
        plt.tight_layout()
        out = OUT_DIR / f"mean_{chan}.png"
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)

    print(f"saved {len(common_chans)} plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
