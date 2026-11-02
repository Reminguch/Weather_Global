#!/usr/bin/env python3
"""Validate native forecasts and overlay the sampled reproduction on Figure 2a."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analyze_models.eval_graphcast_fig2a import paper_rmse, read, sha, write_json
from scripts.experiments.frozen_runtime import verify_source
from scripts.analyze_models.graphcast_fig2a_data import LocalERA5, array_sha


def groups(indices):
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)


def digitize(image_path):
    """Read plotted center lines with axes calibrated from raster tick marks."""
    a = np.asarray(Image.open(image_path).convert("RGB"))
    if a.shape != (741, 1495, 3):
        raise ValueError("Published raster geometry changed")
    xhits = np.flatnonzero(np.all(a[421, 110:470] < 130, axis=1)) + 110
    xticks = np.array([g.mean() for g in groups(xhits)])
    yhits = np.flatnonzero(np.all(a[40:415, 83] < 130, axis=1)) + 40
    yticks = np.array([g.mean() for g in groups(yhits)])
    if len(xticks) != 10 or len(yticks) != 9:
        raise ValueError("Could not identify the published axes")
    xfit = np.polyfit(np.arange(1, 11), xticks, 1)
    yfit = np.polyfit(yticks, np.arange(800, -1, -100), 1)
    color = np.linalg.norm(a.astype(float) - [60, 140, 195], axis=2) < 12
    black = np.all(a < 40, axis=2)
    rows = []
    for day in np.arange(.5, 10.01, .5):
        x = int(round(np.polyval(xfit, day)))
        # Avoid the vertical changeover marker when extracting the HRES line.
        xs = [x - 2, x + 2] if abs(day - 3.5) < 1e-9 else [x]
        gc_y, hres_y = [], []
        for col in xs:
            hits = np.flatnonzero(color[45:414, col]) + 45
            if not len(hits):
                raise ValueError(f"Missing blue curve at day {day}")
            center = float(np.median(groups(hits)[0]))  # excludes the lower legend
            gc_y.append(center)
            lo, hi = max(45, int(center) - 65), min(414, int(center))
            candidates = np.flatnonzero(black[lo:hi, col]) + lo
            if not len(candidates):
                raise ValueError(f"Missing HRES curve at day {day}")
            hres_y.append(float(np.median(groups(candidates)[-1])))
        rows.append({"lead_hours": int(day * 24),
                     "paper_graphcast_rmse": float(np.polyval(yfit, np.mean(gc_y))),
                     "paper_hres_rmse": float(np.polyval(yfit, np.mean(hres_y)))})
    return {"source_sha256": sha(image_path), "rows": rows,
            "method": "raster center lines, calibrated from integer axis ticks",
            "x_tick_centers": xticks.tolist(), "y_tick_centers": yticks.tolist(),
            "one_vertical_pixel_rmse": abs(float(yfit[0])),
            "limitation": "Approximate digitized references, not original numerical scores or rerun HRES"}


def independent_area_weights(latitude):
    latitude = np.asarray(latitude, dtype=np.float64)
    midpoints = (latitude[:-1] + latitude[1:]) / 2
    bounds = np.r_[-90., midpoints, 90.]
    area = np.diff(np.sin(np.deg2rad(bounds)))
    return area / area.mean()


def report(workflow, references_only=False):
    p = read(workflow / "protocol.json")
    reference = digitize(workflow / "references/paper_figure2.png")
    write_json(workflow / "references/figure2a_digitized.json", reference)
    if references_only:
        print(json.dumps(reference, indent=2))
        return
    verify_source(Path(p["eval_source"]))
    protocol_sha = sha(workflow / "protocol.json")
    rows, all_mse, validation = [], [], []
    expected_lat = np.arange(-90, 90.25, .25)
    expected_lon = np.arange(0, 360, .25)
    weights = independent_area_weights(expected_lat)
    local = LocalERA5(p["prepared_root"], p["prepared_identity"]) if "prepared_root" in p else None
    for initialization in p["initialization_times"]:
        tag = initialization.replace("-", "").replace(":", "")
        result_path = workflow / "results" / f"{tag}.json"
        r = read(result_path)
        if (r.get("status") != "complete" or r["initialization_time"] != initialization
                or r["protocol_sha256"] != protocol_sha or r["lead_hours"] != p["lead_hours"]
                or r["checkpoint_sha256"] != p["checkpoint"]["sha256"]):
            raise ValueError("Result identity or coverage mismatch")
        if sha(r["forecast_path"]) != r["forecast_sha256"]:
            raise ValueError("Saved forecast changed")
        recomputed = []
        with xr.open_dataset(r["forecast_path"]) as pred:
            np.testing.assert_array_equal(pred.lat, expected_lat)
            np.testing.assert_array_equal(pred.lon, expected_lon)
            np.testing.assert_array_equal(pred.lead_hours, p["lead_hours"])
            for index, lead in enumerate(p["lead_hours"]):
                valid = np.datetime64(initialization.removesuffix("Z")) + np.timedelta64(lead, "h")
                valid_tag = np.datetime_as_string(valid, unit="s").replace("-", "").replace(":", "") + "Z"
                if local is not None:
                    values = local.truth(valid)
                    if array_sha(values) != r["truth_sha256"][index]:
                        raise ValueError("Local truth changed after evaluation")
                else:
                    truth_path = Path(p["data_root"]) / "truth" / f"{valid_tag}.nc"
                    if sha(truth_path) != r["truth_sha256"][index]:
                        raise ValueError("Truth changed after evaluation")
                    with xr.open_dataset(truth_path) as truth:
                        np.testing.assert_array_equal(truth.lat, expected_lat)
                        np.testing.assert_array_equal(truth.lon, expected_lon)
                        values = truth.Z500.values
                error = pred.Z500.isel(lead_hours=index).values.astype(np.float64) - values.astype(np.float64)
                recomputed.append(float(np.sqrt(np.mean(error**2 * weights[:, None]))))
                rows.append({"initialization_time": initialization, "lead_hours": lead,
                             "spatial_mse": r["mse"][index], "spatial_rmse": r["rmse"][index]})
        np.testing.assert_allclose(recomputed, r["rmse"], rtol=3e-6, atol=1e-4)
        validation.append({"initialization_time": initialization,
                           "max_relative_metric_difference": float(np.max(np.abs(np.asarray(recomputed) / r["rmse"] - 1))),
                           "result_sha256": sha(result_path), "forecast_sha256": r["forecast_sha256"]})
        all_mse.append(r["mse"])
    mse = np.asarray(all_mse, dtype=np.float32)
    mean = paper_rmse(mse)
    per_start_rmse = np.sqrt(mse).astype(np.float64)
    n_dates = len(all_mse) // 2
    annual = len(all_mse) == p["paper_full_year_count"]
    date_means = per_start_rmse.reshape(n_dates, 2, 20).mean(axis=1)
    rng = np.random.default_rng(20261019)
    if annual:
        # Preserve within-day pairing and two weeks of serial dependence.
        blocks = rng.integers(0, n_dates, size=(10000, int(np.ceil(n_dates / 14))))
        indices = ((blocks[..., None] + np.arange(14)) % n_dates).reshape(10000, -1)[:, :n_dates]
        draws = np.stack([date_means[i].mean(axis=0) for i in indices])
    else:
        draws = date_means[rng.integers(0, n_dates, size=(10000, n_dates))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    pub = np.array([r["paper_graphcast_rmse"] for r in reference["rows"]])
    hres = np.array([r["paper_hres_rmse"] for r in reference["rows"]])
    table = Path(p["workspace"]) / "plots/analyze_models/data/resolution_eval" / workflow.name
    images = Path(p["workspace"]) / "plots/analyze_models/images/resolution_eval" / workflow.name
    table.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    with (table / "per_initialization.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"status": "complete_annual_graphcast_curve" if annual else "complete_sampled_reproduction", "n_starts": len(all_mse),
               "protocol_sha256": protocol_sha, "checkpoint_sha256": p["checkpoint"]["sha256"],
               "eval_source_manifest_sha256": sha(Path(p["eval_source"]) / "snapshot.json"),
               "lead_hours": p["lead_hours"], "mean_spatial_rmse": mean.tolist(),
               "date_cluster_bootstrap_95_lower": lo.tolist(), "date_cluster_bootstrap_95_upper": hi.tolist(),
               "paper_graphcast_rmse_approx": pub.tolist(), "paper_hres_rmse_approx": hres.tolist(),
               "relative_to_paper_graphcast_pct_approx": (100 * (mean / pub - 1)).tolist(),
               "pooled_mse_rmse_secondary": np.sqrt(mse.mean(axis=0)).tolist(),
               "units": "m2 s-2", "spatial_grid": "native 0.25 degree",
               "aggregation": p["verification"]["aggregation"],
               "limitations": ([] if annual else [f"{len(all_mse)} starts, not the full 730-start paper sample"]) + [
                               "Paper curves are approximate raster digitizations",
                               "Bootstrap bands describe the sampled dates and do not reproduce paper AR(2) intervals",
                               "HRES reference is digitized, not recomputed"],
               "validation": validation, "reference": reference}
    write_json(workflow / "report/summary.json", summary)
    write_json(table / "summary.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 180})
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    days = np.array(p["lead_hours"]) / 24
    ax.plot(days, hres, color="black", linewidth=1.7, label="Paper HRES (digitized)")
    ax.plot(days, pub, color="#3c8cc3", linestyle="--", linewidth=1.7, label="Paper GraphCast (digitized)")
    ax.fill_between(days, lo, hi, color="#d66c22", alpha=.16)
    ax.plot(days, mean, color="#b85313", marker="o", markersize=3, linewidth=1.8,
            label=f"Public GraphCast rerun ({len(all_mse)} starts)")
    ax.axvline(3.5, color=".45", linestyle="--", linewidth=.8)
    ax.set(xlabel="Forecast lead time (days)", ylabel="Z500 RMSE (m² s⁻²)",
           xlim=(.4, 10.1), ylim=(0, max(850, float(hi.max()) * 1.06)), xticks=np.arange(1, 11),
           title="Figure 2a reproduction · native 0.25° · " + ("full 2018" if annual else "sampled 2018"))
    ax.legend(frameon=False, loc="upper left")
    fig.get_layout_engine().set(rect=(0, .09, 1, .90))
    fig.text(.5, .015, "Official 37-level checkpoint · mean of per-forecast spatial RMSE\n"
             "Shading: approximate bootstrap; published curves digitized from paper.",
             ha="center", fontsize=8, color=".3")
    for suffix in ("png", "pdf"):
        fig.savefig(images / f"figure2a_reproduction.{suffix}")
    plt.close(fig)
    text = ["# GraphCast Figure 2a reproduction", "", p["scope"], "",
            f"Checkpoint SHA256: `{p['checkpoint']['sha256']}`.", "",
            "| Lead | Our mean spatial RMSE | Paper GC, approximate | Difference |",
            "| --- | ---: | ---: | ---: |"]
    for day in (1, 2, 3, 5, 10):
        i = day * 2 - 1
        text.append(f"| {day} days | {mean[i]:.3f} | {pub[i]:.3f} | {100 * (mean[i] / pub[i] - 1):+.2f}% |")
    text += ["", *summary["limitations"], "", f"All {len(rows)} scores were independently recomputed from saved native predictions and truth."]
    (workflow / "report/report.md").write_text("\n".join(text) + "\n")
    print(images / "figure2a_reproduction.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--references-only", action="store_true")
    args = parser.parse_args()
    report(args.workflow.resolve(), args.references_only)
