#!/usr/bin/env python3
"""Render an already independently validated Figure 2a summary."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analyze_models.eval_graphcast_fig2a import read, sha, write_json


def render(workflow):
    p = read(workflow / "protocol.json")
    s = read(workflow / "report/summary.json")
    if s["protocol_sha256"] != sha(workflow / "protocol.json"):
        raise ValueError("Summary belongs to another protocol")
    if s["status"] not in ("complete_sampled_reproduction", "complete_annual_graphcast_curve"):
        raise ValueError("Summary has not passed independent validation")
    if s["n_starts"] != len(p["initialization_times"]) or len(s["validation"]) != s["n_starts"]:
        raise ValueError("Incomplete forecast coverage")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 180})
    images = Path(p["workspace"]) / "plots/analyze_models/images/resolution_eval" / workflow.name
    images.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    days = np.asarray(s["lead_hours"]) / 24
    lo, hi = np.asarray(s["date_cluster_bootstrap_95_lower"]), np.asarray(s["date_cluster_bootstrap_95_upper"])
    ax.plot(days, s["paper_hres_rmse_approx"], color="black", linewidth=1.7, label="Paper HRES (digitized)")
    ax.plot(days, s["paper_graphcast_rmse_approx"], color="#3c8cc3", linestyle="--", linewidth=1.7,
            label="Paper GraphCast (digitized)")
    ax.fill_between(days, lo, hi, color="#d66c22", alpha=.16)
    ax.plot(days, s["mean_spatial_rmse"], color="#b85313", marker="o", markersize=3, linewidth=1.8,
            label=f"Public GraphCast rerun ({s['n_starts']} starts)")
    ax.axvline(3.5, color=".45", linestyle="--", linewidth=.8)
    annual = s["n_starts"] == p["paper_full_year_count"]
    ax.set(xlabel="Forecast lead time (days)", ylabel="Z500 RMSE (m² s⁻²)",
           xlim=(.4, 10.1), ylim=(0, max(850, float(hi.max()) * 1.06)), xticks=np.arange(1, 11),
           title="Figure 2a reproduction · native 0.25° · " + ("full 2018" if annual else "sampled 2018"))
    ax.legend(frameon=False, loc="upper left")
    # Constrained-layout rect is (left, bottom, width, height), not corners.
    fig.get_layout_engine().set(rect=(0, .09, 1, .90))
    fig.text(.5, .015, "Official 37-level checkpoint · mean of per-forecast spatial RMSE\n"
             "Shading: approximate bootstrap; published curves digitized from paper.",
             ha="center", fontsize=8, color=".3")
    for suffix in ("png", "pdf"):
        fig.savefig(images / f"figure2a_reproduction.{suffix}")
    plt.close(fig)
    write_json(workflow / "report/plot_receipt.json", {
        "summary_sha256": sha(workflow / "report/summary.json"),
        "plot_script_sha256": sha(Path(__file__)),
        "source_manifest_sha256": sha(ROOT / "snapshot.json") if (ROOT / "snapshot.json").exists() else None,
        "png_sha256": sha(images / "figure2a_reproduction.png"),
        "pdf_sha256": sha(images / "figure2a_reproduction.pdf"),
        "note": "Layout correction only; scores and confidence intervals are unchanged",
    })
    print(images / "figure2a_reproduction.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    render(parser.parse_args().workflow.resolve())
