"""Plot saved AMSE campaign selection/confirmation records; no inference.

Usage: source scripts/graphcast_env.sh && python scripts/analyze_models/plot_v24_amse_regime.py \
    --report /path/to/joint_selection.json --out-dir /path/to/images
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#9A7315", "#6B5B95", "#666666"]
FAMILY_LABELS = {"mature_mse": "MSE initialization", "mse": "MSE continuation",
                 "spectral_mse": "Spectral MSE control", "amse_l025": "AMSE λ=0.25",
                 "amse_l050": "AMSE λ=0.5", "full_amse": "Full AMSE",
                 "amse_lambda0p1": "AMSE λ=0.1", "mse_trajectory": "MSE trajectory",
                 "confirmation_mse": "MSE confirmation", "confirmation_amse": "AMSE confirmation"}
METRICS = {
    "physical_rmse": ("rmse_ratio_vs_gc", "Physical RMSE reduction vs Original GC (%)", lambda x: 100*(1-x)),
    "power_error": ("power_error_ratio_vs_gc", "Relative power-error reduction vs Original GC (%)", lambda x: 100*(1-x)),
    "coherence": ("coherence_change_vs_mse", "Coherence change vs MSE (percentage points)", lambda x: 100*x),
}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 13, "axes.labelsize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.facecolor": "white"})


def identity(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def record_label(record):
    label = record.get("label", record["id"])
    if record["id"].startswith("confirmation_"):
        label = FAMILY_LABELS.get(record.get("family", record.get("recipe_id", record["id"])), label)
    details = []
    if "swa" in record["id"].lower():
        details.append("SWA")
    if record.get("seed") is not None:
        details.append(f"seed {record['seed']}")
    if record.get("step") is not None:
        details.append(f"update {record['step']}")
    if record.get("n_seeds") is not None:
        details.append(f"{record['n_seeds']} seeds")
    if not details and record["id"] != label:
        details.append(record["id"])
    return label + (" · " + " · ".join(details) if details else "")


def feature_coordinates(records):
    first = records[0]["score"]["per_feature_lead"]
    coordinates = [(r["feature"], int(r["day"])) for r in first]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("Repeated feature/lead rows")
    for record in records:
        actual = [(r["feature"], int(r["day"])) for r in record["score"]["per_feature_lead"]]
        if actual != coordinates:
            raise ValueError(f"Feature/lead ordering differs for {record['id']}")
    return coordinates


def caption(fig, text, y=.025):
    fig.text(.06, y, text, fontsize=9, color="#586572", va="bottom")


def save(fig, stem, output, manifest, **details):
    files = {}
    for extension in ("png", "pdf"):
        path = output/f"{stem}.{extension}"
        fig.savefig(path, dpi=170)
        files[extension] = str(path.resolve())
    plt.close(fig)
    manifest["figures"].append({"name": stem, "files": files, **details})


def percentile_interval(record, metric):
    bootstrap = record.get("bootstrap_vs_gc", {})
    values = bootstrap.get("percentile95", {}).get(metric)
    if bootstrap.get("status") != "complete" or values is None:
        return None
    lower, upper = map(float, values)
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError(f"Invalid confidence interval for {record['id']}/{metric}")
    return 100*(1-upper), 100*(1-lower)


def scatter(report, records, output, manifest, *, stem="joint_rmse_power", recipes=False):
    fig, ax = plt.subplots(figsize=(11.5, 8.7))
    phase = "Confirmation" if report["evaluation_role"] == "confirmation" else "Exploratory comparison"
    title = "Joint forecast-error and spectral-power comparison"
    if recipes:
        title = "Confirmation across independently trained seeds"
    fig.suptitle(title, fontsize=18, weight="bold", y=.975)
    fig.text(.5, .935, f"{phase} · both axes use the same evaluation starts · upper right improves both diagnostics",
             ha="center", fontsize=10.5, color="#485460")
    families = list(dict.fromkeys(row.get("family", row.get("recipe_id", row["id"])) for row in records))
    color = {family: COLORS[i % len(COLORS)] for i, family in enumerate(families)}
    selected = {row["id"] for row in report.get("finalists", [])}
    values = np.asarray([[100*(1-row["score"]["mean_rmse_ratio_vs_gc"]),
                          100*(1-row["score"]["mean_power_error_ratio_vs_gc"])] for row in records])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite joint comparison")
    extents = list(values)
    if report["evaluation_role"] == "confirmation":
        for record, point in zip(records, values, strict=True):
            xi = percentile_interval(record, "mean_rmse_ratio")
            yi = percentile_interval(record, "mean_power_error_ratio")
            if xi is not None:
                extents.extend([[xi[0], point[1]], [xi[1], point[1]]])
            if yi is not None:
                extents.extend([[point[0], yi[0]], [point[0], yi[1]]])
    extents = np.asarray(extents)
    xmin, ymin = np.minimum(extents.min(axis=0), 0)
    xmax, ymax = np.maximum(extents.max(axis=0), 0)
    xpad, ypad = max(1., (xmax-xmin)*.12), max(2., (ymax-ymin)*.12)
    ax.set_xlim(xmin-xpad, xmax+xpad)
    ax.set_ylim(ymin-ypad, ymax+ypad)
    ax.add_patch(Rectangle((0, 0), xmax+xpad, ymax+ypad,
                           facecolor="#E5F1E8", alpha=.8, edgecolor="none", zorder=0))
    ax.axhline(0, color="#8A929B", linewidth=1)
    ax.axvline(0, color="#8A929B", linewidth=1)
    ax.grid(color="#E3E7EB", linewidth=.7, alpha=.7)
    ax.set_axisbelow(True)
    mapping = []
    for family in families:
        positions = [i for i, r in enumerate(records)
                     if r.get("family", r.get("recipe_id", r["id"])) == family]
        # These segments connect separately evaluated checkpoints of a training
        # trajectory. They never interpolate forecast fields or measurements.
        if (len(positions) > 1 and all(records[i].get("step") is not None for i in positions)
                and report["evaluation_role"] == "development" and not recipes):
            ordered = sorted(positions, key=lambda i: records[i]["step"])
            ax.plot(values[ordered, 0], values[ordered, 1], color=color[family], alpha=.3, linewidth=1, zorder=1)
    for i, (record, point) in enumerate(zip(records, values, strict=True), 1):
        family = record.get("family", record.get("recipe_id", record["id"]))
        chosen = record["id"] in selected
        marker = "*" if chosen else "D" if record["id"] == report["canonical_mse_id"] else "o"
        ax.scatter(*point, color=color[family], marker=marker, s=145 if chosen else 65,
                   edgecolors="#202A34" if record["score"].get("eligible") else "white", linewidth=1,
                   zorder=5)
        # Always show every available confirmation interval, not just favorable ones.
        if report["evaluation_role"] == "confirmation":
            xinterval = percentile_interval(record, "mean_rmse_ratio")
            yinterval = percentile_interval(record, "mean_power_error_ratio")
            if xinterval is not None:
                ax.plot(xinterval, [point[1], point[1]], color=color[family], linewidth=1, alpha=.75, zorder=3)
            if yinterval is not None:
                ax.plot([point[0], point[0]], yinterval, color=color[family], linewidth=1, alpha=.75, zorder=3)
        ax.annotate(str(i), point, xytext=(5, 5), textcoords="offset points", fontsize=8.5, color=color[family])
        mapping.append({"number": i, "id": record["id"], "label": record_label(record),
                        "mean_rmse_reduction_pct": float(point[0]), "power_error_reduction_pct": float(point[1]),
                        "exact_gc_loss_reduction_pct": 100*(1-record["score"]["exact_gc_loss_ratio_vs_gc"]),
                        "eligible": record["score"].get("eligible")})
    ax.scatter(0, 0, marker="X", color="#555E68", s=80, zorder=7)
    ax.annotate("Original GC", (0, 0), xytext=(8, -17), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Mean physical RMSE reduction vs Original GC (%)")
    ax.set_ylabel("Mean relative power-error reduction vs Original GC (%)")
    legend = [Line2D([], [], marker="o", linestyle="", color=color[f],
                     label=FAMILY_LABELS.get(f, f.replace("_", " "))) for f in families]
    ax.legend(handles=legend, frameon=False, loc="best", fontsize=9)
    caption(fig, "RMSE averages physical-field ratios across the prespecified fields/leads; power error averages relative spectral errors.", .091)
    table_name = "confirmation_recipe_candidate_metrics.csv" if recipes else "candidate_metrics.csv"
    caption(fig, f"Aggregate eligibility also requires lower exact GraphCast loss and the safety guard. Numbers map to {table_name}.", .067)
    if recipes:
        caption(fig, "Seed summaries average forecast metrics from separate models. No forecast fields are averaged; these are not ensemble-mean scores.", .043)
    elif report["evaluation_role"] == "confirmation":
        caption(fig, "Lines through confirmation points show available paired 95% intervals. Each point remains an independently evaluated model.", .043)
    else:
        caption(fig, "Checkpoint comparisons are exploratory. Connecting lines trace training checkpoints; stars mark selected finalists.", .043)
    fig.subplots_adjust(left=.10, right=.975, top=.88, bottom=.19)
    save(fig, stem, output, manifest, candidates=mapping, reference="canonical Original GC",
         metric_aggregation="equal feature/selected-lead RMSE ratios and equal spectral power-error bins",
         no_forecast_averaging=True)
    return mapping


def heatmaps(report, records, output, manifest, *, prefix="feature"):
    coordinates = feature_coordinates(records)
    labels = [record_label(record) for record in records]
    for metric, (key, label, transform) in METRICS.items():
        values = np.array([[transform(float(row[key])) for row in record["score"]["per_feature_lead"]]
                           for record in records])
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite feature heatmap: {key}")
        bound = max(float(np.max(np.abs(values))), .1 if metric == "coherence" else 1.)
        fig, ax = plt.subplots(figsize=(17.8, max(5.0, .37*len(records)+2.5)))
        fig.suptitle(label, fontsize=17, weight="bold", y=.97)
        fig.text(.5, .919, "Each cell is one physical field and forecast lead · positive values favor the candidate",
                 ha="center", fontsize=10, color="#485460")
        im = ax.imshow(values, aspect="auto", cmap="RdBu", vmin=-bound, vmax=bound, interpolation="nearest")
        ax.set_yticks(np.arange(len(records)), labels, fontsize=8.5)
        ax.set_xticks(np.arange(len(coordinates)), [f"{feature}\n{day}d" for feature, day in coordinates], fontsize=7.3)
        ax.tick_params(length=0)
        for col in range(1, len(coordinates)):
            if coordinates[col][0] != coordinates[col-1][0]:
                ax.axvline(col-.5, color="white", linewidth=1.5)
        if len(records) <= 10:
            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    value = values[row, col]
                    ax.text(col, row, f"{value:+.1f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(value) > bound*.55 else "#1F2933")
        bar = fig.colorbar(im, ax=ax, fraction=.022, pad=.014)
        bar.set_label("Percentage points" if metric == "coherence" else "Reduction (%)", fontsize=9)
        caption(fig, "Global RMSE includes zonal means; zonal spectra/coherence use 30–60° latitude in both hemispheres. The spatial scopes differ.", .049)
        caption(fig, "An aggregate win does not imply that every cell improves. All evaluated candidates are retained, including safety failures.", .023)
        fig.subplots_adjust(left=.24, right=.945, top=.857, bottom=.15)
        save(fig, f"{prefix}_{metric}", output, manifest,
             ids=[row["id"] for row in records], metric=key, coordinates=coordinates,
             color_limits=[-bound, bound])


def export_tables(records, mapping, output, prefix=""):
    columns = ["number", "id", "label", "mean_rmse_reduction_pct", "power_error_reduction_pct", "exact_gc_loss_reduction_pct", "eligible"]
    with (output/f"{prefix}candidate_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(mapping)
    rows = [{"id": record["id"], "label": record_label(record), **row}
            for record in records for row in record["score"]["per_feature_lead"]]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output/f"{prefix}feature_lead_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    intervals = []
    aggregate = []
    for record in records:
        for reference in ("gc", "mse"):
            bootstrap = record.get(f"bootstrap_vs_{reference}", {})
            for row in bootstrap.get("per_feature_lead95", []):
                for metric in ("rmse_ratio", "power_error_ratio", "coherence_change"):
                    lower, upper = row[metric]
                    intervals.append({"id": record["id"], "reference": reference,
                                      "feature": row["feature"], "day": row["day"], "metric": metric,
                                      "lower95": lower, "upper95": upper,
                                      "replicates": bootstrap.get("replicates"),
                                      "grouping": bootstrap.get("grouping"), "caveat": bootstrap.get("caveat")})
            for metric, (lower, upper) in bootstrap.get("percentile95", {}).items():
                aggregate.append({"id": record["id"], "reference": reference, "metric": metric,
                                  "lower95": lower, "upper95": upper,
                                  "replicates": bootstrap.get("replicates"),
                                  "grouping": bootstrap.get("grouping"), "caveat": bootstrap.get("caveat")})
    for name, values in (("paired_feature_intervals", intervals), ("paired_aggregate_intervals", aggregate)):
        if values:
            with (output/f"{prefix}{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)


def exact_loss(report, records, output, manifest, *, stem="exact_graphcast_loss", recipes=False):
    """Canonical headline loss retains its own exact aggregation and intervals."""
    values = np.array([100*(1-row["score"]["exact_gc_loss_ratio_vs_gc"]) for row in records])
    fig, ax = plt.subplots(figsize=(11.5, max(5., .38*len(records)+2.3)))
    fig.suptitle("Exact original GraphCast loss reduction", fontsize=17, weight="bold", y=.96)
    ax.axvline(0, color="#8A929B", linewidth=1)
    ax.barh(np.arange(len(records)), values, color=[COLORS[i % len(COLORS)] for i in range(len(records))], alpha=.8)
    if report["evaluation_role"] == "confirmation":
        for i, record in enumerate(records):
            interval = percentile_interval(record, "exact_gc_loss_ratio")
            if interval is not None:
                ax.plot(interval, [i, i], color="#202A34", linewidth=1.3)
                ax.scatter(interval, [i, i], marker="|", color="#202A34", s=35)
    ax.set_yticks(np.arange(len(records)), [record_label(row) for row in records], fontsize=8.5)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E3E7EB", linewidth=.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Reduction vs common Original GC (%) · higher is better")
    caption(fig, "Exact GraphCast weighted losses are pooled across the same starts and all 40 leads before taking the ratio.", .078)
    if recipes:
        text = "Seed summaries average separate-model loss ratios; intervals resample seeds and paired seasonal weather blocks. No field averaging."
    elif report["evaluation_role"] == "confirmation":
        text = "Intervals are paired 95% seasonal-block bootstrap intervals. Each bar is one independently evaluated trained model."
    else:
        text = "These exploratory comparisons are distinct from equal-field physical RMSE and selected-lead power diagnostics."
    caption(fig, text, .032)
    fig.subplots_adjust(left=.35, right=.975, top=.87, bottom=.21)
    save(fig, stem, output, manifest, ids=[row["id"] for row in records],
         metric="exact_gc_loss_ratio_vs_gc", reference="canonical Original GC", no_forecast_averaging=True)


def feature_intervals(records, output, manifest):
    """All prespecified feature/lead intervals for the independently trained recipes."""
    coordinates = feature_coordinates(records)
    metrics = [("rmse_ratio", "rmse_ratio", "Physical RMSE reduction (%)", lambda x: 100*(1-x)),
               ("power_error_ratio", "power_error_ratio", "Relative power-error reduction (%)", lambda x: 100*(1-x)),
               ("coherence_change", "coherence_change", "Coherence change (percentage points)", lambda x: 100*x)]
    for reference in ("gc", "mse"):
        reference_label = "Original GC" if reference == "gc" else "matched MSE seed"
        available = [row for row in records if row.get(f"bootstrap_vs_{reference}", {}).get("status") == "complete"]
        if not available:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(16, 13.3), sharey=True)
        fig.suptitle(f"Confirmation feature/lead intervals vs {reference_label}", fontsize=18, weight="bold", y=.979)
        fig.text(.5, .947, "Means of separate-model metrics across training seeds · every prespecified field/lead retained",
                 ha="center", fontsize=11, color="#485460")
        offsets = np.linspace(-.18, .18, len(available)) if len(available) > 1 else [0.]
        for index, record in enumerate(available):
            interval = record[f"bootstrap_vs_{reference}"]
            rows = {(row["feature"], int(row["day"])): row for row in interval["per_feature_lead95"]}
            if set(rows) != set(coordinates):
                raise ValueError(f"Incomplete feature/lead intervals for {record['id']}/{reference}")
            point_rows = {(row["feature"], int(row["day"])): row for row in record["score"]["per_feature_lead"]}
            y = np.arange(len(coordinates))+offsets[index]
            for ax, (interval_key, score_key, xlabel, transform) in zip(axes, metrics, strict=True):
                points = [transform(point_rows[key][f"{score_key}_vs_{reference}"]) for key in coordinates]
                bounds = np.array([sorted(transform(float(value)) for value in rows[key][interval_key]) for key in coordinates])
                if not np.isfinite(bounds).all():
                    raise ValueError(f"Nonfinite feature/lead intervals for {record['id']}/{reference}")
                ax.hlines(y, bounds[:, 0], bounds[:, 1], color=COLORS[index % len(COLORS)], linewidth=1.15, alpha=.8)
                ax.scatter(points, y, color=COLORS[index % len(COLORS)], s=15, zorder=3)
                ax.set_xlabel(xlabel, fontsize=10)
        for ax in axes:
            ax.axvline(0, color="#65727F", linewidth=1)
            ax.grid(axis="x", color="#E3E7EB", linewidth=.7)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0)
            for row in range(1, len(coordinates)):
                if coordinates[row][0] != coordinates[row-1][0]:
                    ax.axhline(row-.5, color="#D3D9DF", linewidth=.75)
        axes[0].set_yticks(np.arange(len(coordinates)), [f"{field} · day {day}" for field, day in coordinates], fontsize=8.5)
        axes[0].set_ylim(len(coordinates)-.4, -.6)
        legend = [Line2D([], [], color=COLORS[i % len(COLORS)], marker="o", linewidth=1,
                         label=f"{row['label']} ({row.get('n_seeds', '?')} seeds)") for i, row in enumerate(available)]
        fig.legend(handles=legend, frameon=False, loc="lower center", bbox_to_anchor=(.5, .055), ncol=len(available))
        caption(fig, "Paired 95% percentile intervals resample seeds and shared seasonal date blocks. Three seeds give limited information about training variability.", .036)
        caption(fig, "Intervals are pointwise, without multiplicity correction. Global RMSE and midlatitude zonal spectral diagnostics use different spatial scopes.", .018)
        fig.subplots_adjust(left=.105, right=.975, top=.915, bottom=.13, wspace=.10)
        save(fig, f"confirmation_recipe_feature_intervals_vs_{reference}", output, manifest,
             ids=[row["id"] for row in available], reference=reference_label, coordinates=coordinates,
             interval="pointwise paired hierarchical seed/seasonal-block 95% percentile", no_forecast_averaging=True)


def render(report_path, out_dir):
    report_path, out_dir = Path(report_path), Path(out_dir)
    report = json.loads(report_path.read_text())
    if report.get("evaluation_role") not in ("development", "confirmation"):
        raise ValueError("Unknown evaluation role")
    records = report["records"]
    if not records:
        raise ValueError("No evaluated candidates to plot")
    out_dir.mkdir(parents=True, exist_ok=True)
    style()
    manifest = {"source": identity(report_path), "status": report["status"],
                "evaluation_role": report["evaluation_role"], "figures": [],
                "no_forecast_averaging": True,
                "scope": report.get("scope"), "selection_policy": report.get("selection_policy")}
    mapping = scatter(report, records, out_dir, manifest)
    exact_loss(report, records, out_dir, manifest)
    heatmaps(report, records, out_dir, manifest)
    export_tables(records, mapping, out_dir)
    recipes = report.get("recipe_summaries", [])
    if isinstance(recipes, dict):
        recipes = list(recipes.values())
    if recipes:
        mapping = scatter(report, recipes, out_dir, manifest, stem="confirmation_recipe_joint", recipes=True)
        exact_loss(report, recipes, out_dir, manifest, stem="confirmation_recipe_exact_graphcast_loss", recipes=True)
        heatmaps(report, recipes, out_dir, manifest, prefix="confirmation_recipe")
        export_tables(recipes, mapping, out_dir, prefix="confirmation_recipe_")
        feature_intervals(recipes, out_dir, manifest)
    manifest["tables"] = [str(path.resolve()) for path in sorted(out_dir.glob("*.csv"))]
    (out_dir/"figures_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(f"Saved {len(manifest['figures'])} campaign figures as PNG/PDF to {out_dir}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    render(args.report, args.out_dir)


if __name__ == "__main__":
    main()
