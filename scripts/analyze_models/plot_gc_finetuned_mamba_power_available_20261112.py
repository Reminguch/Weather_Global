#!/usr/bin/env python3
"""Plot the available matched day-10 spectra, explicitly as a 2022 pilot."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analyze_models.report_v24_raw_checkpoint_pilot import FIELDS, divide

NAME = "gc_finetuned_mamba_power_available_20261112"
SOURCE = ROOT / "plots/analyze_models/data/resolution_eval/v24_raw_checkpoint_pilot_20261110"
DATA = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
IMAGES = ROOT / "plots/analyze_models/images/resolution_eval" / NAME
MODELS = {
    "public_gc": ("Original GraphCast", "#555555", "--"),
    "gcft_s22_l000": ("GC fine-tuned · raw 1,250 updates", "#d55e00", "-"),
    "M1_raw": ("GC–Mamba M1 · raw 500 + 2,000 updates", "#0072b2", "-"),
}


def pinned(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    report_path = SOURCE / "report.json"
    spectra_path = SOURCE / "pooled_spectra.npz"
    report = json.loads(report_path.read_text())
    if report["status"] != "complete":
        raise ValueError("Source pilot is incomplete")
    if pinned(spectra_path)["sha256"] != report["pooled_spectra"]["sha256"]:
        raise ValueError("Source spectra checksum mismatch")
    manifest_path = Path(report["sources"][0]["path"])
    if pinned(manifest_path)["sha256"] != report["sources"][0]["sha256"]:
        raise ValueError("Source manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    with np.load(spectra_path, allow_pickle=False) as saved:
        metadata = json.loads(saved["metadata"].item())
        channels = [tuple(channel) for channel in metadata["layout"]["channels"]]
        model_ids = saved["models"].tolist()
        degree = saved["degree"][1:]
        lead = saved["lead_hours"].tolist().index(240)
        truth = saved["truth_power"][lead, :, 1:]
        power = {identity: saved["prediction_power"][model_ids.index(identity), lead, :, 1:]
                 for identity in MODELS}
    if metadata["initialization_times"] != report["initialization_times"]:
        raise ValueError("Spectral cohort differs from source report")
    if report["count"] != 4 or not np.array_equal(degree, np.arange(1, 180)):
        raise ValueError("Unexpected pilot cohort or degree range")
    if not all(np.isfinite(values).all() for values in [truth, *power.values()]):
        raise ValueError("Nonfinite source spectral values")
    DATA.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    rows = []
    for variable, level, label, units in FIELDS:
        channel = channels.index((variable, level))
        for identity, values in {"ERA5": truth, **power}.items():
            ratios = divide(values[channel], truth[channel])
            for index, ell in enumerate(degree):
                rows.append({"model": identity, "variable": variable, "level_hpa": level,
                             "lead_hours": 240, "spherical_degree": int(ell),
                             "physical_power": float(values[channel, index]),
                             "power_units": units, "power_ratio_to_ERA5": float(ratios[index])})
    with (DATA / "day10_spectra.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    files = []
    for ratio in (False, True):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (variable, level, label, units) in zip(axes, FIELDS):
            channel = channels.index((variable, level))
            reference = np.ones_like(degree) if ratio else truth[channel]
            ax.plot(degree, reference, color="black", linewidth=2,
                    label="ERA5 truth = 1" if ratio else "ERA5 truth")
            for identity, (legend, color, linestyle) in MODELS.items():
                values = power[identity][channel]
                ax.plot(degree, divide(values, truth[channel]) if ratio else values,
                        label=legend, color=color, linestyle=linestyle, linewidth=1.8)
            ax.set(xscale="log", yscale="log", xlim=(1, 179), title=label,
                   xlabel="Spherical harmonic degree",
                   ylabel="Forecast / ERA5 power (log scale)" if ratio else f"Physical power ({units})")
            ax.grid(alpha=.2, which="major")
        title = "Day-10 power relative to ERA5" if ratio else "Day-10 physical power spectra"
        fig.suptitle(title + " · 4-start 2022 pilot", fontsize=16, y=.99)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
                   bbox_to_anchor=(.5, .915), ncols=2, fontsize=10, frameon=False)
        fig.text(.5, .015, "Della pilot: GCFT λ = 0, peak LR 3e−7. Powers pooled over the same four dates. "
                 "Separate from the LH 2023 / GCFT 2,500-update comparison.",
                 ha="center", fontsize=9, color="#444444")
        fig.tight_layout(rect=(0, .065, 1, .80))
        name = "power_ratio_day10" if ratio else "power_spectrum_day10"
        for suffix in ("png", "pdf"):
            output = IMAGES / f"{name}.{suffix}"
            fig.savefig(output, dpi=180, bbox_inches="tight")
            files.append(pinned(output))
        plt.close(fig)

    selected = [model for model in manifest["models"] if model["id"] in MODELS]
    provenance = {
        "purpose": "Matched available physical power spectra; no new forecasts or training",
        "created_date": "2026-11-12", "cohort": report["initialization_times"],
        "lead_hours": 240, "degrees": [1, 179], "spectral_layout": metadata["layout"],
        "aggregation": "Use source powers already pooled across the four forecast dates, then form forecast/ERA5 ratios",
        "models": selected, "public_gc_checkpoint": manifest["baseline"],
        "public_gc_forecast_source": report["baseline_reference"],
        "sources": [pinned(report_path), pinned(spectra_path), pinned(manifest_path)],
        "code": pinned(Path(__file__).resolve()), "data": pinned(DATA / "day10_spectra.csv"),
        "figures": files, "row_count": len(rows),
        "numerical_reference_differences": [item for item in report["numerical_reference_differences"]
                                             if item["model"] in MODELS],
        "noise_scope": report["noise_scope"],
        "limits": "Four development starts, one historical M1 training seed, and intermediate raw GCFT1250. "
                  "M1 raw main2000 follows500 warm-up. Different training budgets and histories. "
                  "These are not the LH2023 local GCFT2500 checkpoints. No confidence intervals for these pooled spectra. "
                  "Finite numerical reference differences remain warnings, not gates.",
    }
    (DATA / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"figures": [item["path"] for item in files], "csv_rows": len(rows),
                      "provenance": str((DATA / "provenance.json").relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
