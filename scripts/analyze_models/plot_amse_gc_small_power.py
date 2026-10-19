"""Plot saved, paired full-AMSE and GraphCast Small zonal power spectra."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "res1_amse_pair_20260919"
MODEL = "amse_swa_step00500-02000"
VARIABLE = "T850"
LEADS = (24, 72, 168, 240)
PREFIX = "amse_lambda1_vs_gc_small_power_T850_swa"


def main():
    data_dir = ROOT / "plots/analyze_models/data/resolution_eval" / EXPERIMENT
    image_dir = ROOT / "plots/analyze_models/images/resolution_eval" / EXPERIMENT
    evaluation_dir = (ROOT / "artifacts/checkpoints/v24_Ilya" / EXPERIMENT
                      / "eval/spectral/amse/swa_step00500-02000")
    spec_path = evaluation_dir / "specification.json"
    spec = json.loads(spec_path.read_text())
    evaluation = json.loads((evaluation_dir / "evaluation.json").read_text())
    assert evaluation["evaluation_status"] == "complete"
    assert evaluation["evaluated_samples"] == spec["n_samples"] == 8
    assert "GraphCast_small" in spec["baseline"]["path"]
    vi = spec["spectral_variables"].index(VARIABLE)
    source = data_dir / "pooled_spectra.npz"
    with np.load(source, allow_pickle=False) as saved:
        wavenumber = saved["wavenumber"].copy()
        assert saved["variables"].tolist() == spec["spectral_variables"]
        assert saved["lead_hours"].tolist() == spec["lead_hours"]
        powers = {branch: saved[f"{MODEL}_{branch}_power"].copy()
                  for branch in ("truth", "baseline", "full")}
    assert all(values.shape == (40, 11, 181) for values in powers.values())
    assert all(np.isfinite(values).all() and (values >= 0).all()
               for values in powers.values())
    # Confirm the plotted baseline is the AMSE evaluation's own GC run.
    raw_paths = sorted(evaluation_dir.glob("*.npz"))
    assert len(raw_paths) == spec["n_samples"]
    raw = {branch: [] for branch in powers}
    for path in raw_paths:
        with np.load(path, allow_pickle=False) as record:
            for branch in raw:
                raw[branch].append(record[f"{branch}_power"])
    for branch in powers:
        np.testing.assert_allclose(powers[branch], np.mean(raw[branch], axis=0),
                                   rtol=1e-12, atol=1e-14)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharex=True, sharey=True)
    styles = [("truth", "ERA5 truth", "#252525", "-", 2.0),
              ("baseline", "GraphCast Small", "#0072B2", "--", 1.9),
              ("full", "GC Small + residual Mamba, AMSE λ=1", "#D55E00", "-", 1.9)]
    rows = []
    for ax, lead in zip(axes.flat, LEADS, strict=True):
        li = spec["lead_hours"].index(lead)
        for branch, label, color, linestyle, linewidth in styles:
            values = powers[branch][li, vi]
            ax.loglog(wavenumber[1:], values[1:], label=label, color=color,
                      linestyle=linestyle, linewidth=linewidth)
            rows.extend((lead / 24, VARIABLE, branch, int(k), float(power))
                        for k, power in zip(wavenumber[1:], values[1:], strict=True))
        ax.set_title(f"Day {lead // 24}", loc="left", fontweight="bold")
        ax.set_xlim(1, 180)
        ax.set_xticks([1, 3, 10, 30, 100, 180], ["1", "3", "10", "30", "100", "180"])
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("Zonal wavenumber k  (higher = finer scales)")
        ax.set_ylabel("Power per zonal mode (K²)")
        ax.grid(which="major", alpha=.22)
        ax.grid(which="minor", alpha=.07)
    fig.suptitle("T850 power spectra: full AMSE vs GraphCast Small",
                 x=.08, y=.975, ha="left", fontsize=18, fontweight="bold")
    fig.text(.08, .932, "Temperature at 850 hPa · 1° grid · Eight matched 2023 starts",
             fontsize=12, color="#505050")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .066),
               ncol=3, frameon=False, fontsize=10.5)
    fig.text(.5, .044, "Zonal Fourier spectra · 30–60° in both hemispheres · Cosine latitude weights · Mean power across starts",
             ha="center", fontsize=9, color="#555555")
    fig.text(.5, .022, "Cold forecasts · SWA at updates 500/1000/1500/2000 · GC baseline paired with AMSE · No spectral smoothing",
             ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=.08, right=.975, top=.87, bottom=.20, hspace=.35, wspace=.20)
    image_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        destination = image_dir / f"{PREFIX}.{extension}"
        fig.savefig(destination, dpi=180, facecolor="white")
        print(destination)
    plt.close(fig)
    with (data_dir / f"{PREFIX}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["lead_days", "variable", "branch", "zonal_wavenumber", "power_K2"])
        writer.writerows(rows)
    metadata = {"source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "specification": str(spec_path), "model": MODEL, "variable": VARIABLE,
                "lead_hours": LEADS, "baseline": spec["baseline"],
                "spectral_metadata": spec["spectral_metadata"],
                "initialization_times": spec["initialization_times"],
                "verified_against_individual_spectra": True}
    (data_dir / f"{PREFIX}.sources.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
