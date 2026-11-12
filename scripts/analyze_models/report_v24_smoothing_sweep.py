#!/usr/bin/env python3
"""Reconstruct exact-grid loss and power tradeoffs from smoothing moments.

The displacement moments retain the native-grid norm. Spherical powers are
never substituted for the original GraphCast objective. Pool moments across
initializations before forming spectral ratios; no setting is fitted here.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WIDTHS = (.25, .5, .75, 1., 1.5, 2., 3., 4., 8.)
BASE_NAMES = [f"fixed_gaussian_{width:g}" for width in WIDTHS] + ["fitted_gaussian", "fitted_band"]
MOMENT_KEYS = ("grid_mse", "prediction_power", "truth_power", "cross_power",
               "delta_grid_cross", "delta_grid_norm", "delta_power",
               "forecast_delta_cross", "delta_truth_cross")


def candidates():
    """One common identity, nine literal widths and ten strengths/profile."""
    result = [{"name": "raw", "series": "raw", "base": None, "alpha": 0., "width_degrees": 0.}]
    for index, width in enumerate(WIDTHS):
        result.append({"name": BASE_NAMES[index], "series": "fixed_gaussian", "base": index,
                       "alpha": 1., "width_degrees": width})
    for base, series in ((9, "fitted_gaussian"), (10, "fitted_band")):
        for tenth in range(1, 11):
            alpha = tenth / 10
            result.append({"name": f"{series}_alpha{alpha:.1f}", "series": series,
                           "base": base, "alpha": alpha, "width_degrees": None})
    return result


def ratio(numerator, denominator):
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    output = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        np.divide(numerator, denominator, out=output, where=denominator > 0)
    if np.isinf(output).any():
        raise ValueError("Nonfinite ratio at a positive denominator")
    return output


def finite_mean(values, axis=None):
    finite = np.isfinite(values)
    return ratio(np.where(finite, values, 0.).sum(axis=axis), finite.sum(axis=axis))


def reconstruct_grid(moments, base=None, alpha=0.):
    if not np.isfinite(alpha):
        raise ValueError("Nonfinite smoothing strength")
    if base is None or alpha == 0:
        return moments["grid_mse"]
    return (moments["grid_mse"] + 2 * alpha * moments["delta_grid_cross"][:, :, base]
            + alpha**2 * moments["delta_grid_norm"][:, :, base])


def reconstruct(moments, base=None, alpha=0.):
    """Return physical grid MSE, forecast power and forecast/truth cross power.

    ``base`` indexes the displacement axis (axis 2); ``alpha`` is a scalar.
    Statistics have lead, model, [displacement], channel, [degree] axes.
    """
    mse = reconstruct_grid(moments, base, alpha)
    power, cross = moments["prediction_power"], moments["cross_power"]
    if base is None or alpha == 0:
        return mse, power, cross
    power = power + 2 * alpha * moments["forecast_delta_cross"][:, :, base] + alpha**2 * moments["delta_power"][:, :, base]
    cross = cross + alpha * moments["delta_truth_cross"][:, :, base]
    return mse, power, cross


def exact_loss(grid_mse, layout):
    scales, weights = np.asarray(layout["scales"]), np.asarray(layout["weights"])
    if not np.isfinite(scales).all() or not np.isfinite(weights).all() or np.any(scales <= 0):
        raise ValueError("Invalid GraphCast channel normalization")
    return np.sum(grid_mse * (weights / scales**2), axis=-1)


def validate_shapes(arrays, metadata, manifest):
    leads, channels, degrees = manifest["target_steps"], len(metadata["layout"]["channels"]), metadata["lmax"] + 1
    prefix = (leads, 2, channels)
    shapes = {"grid_mse": prefix, "prediction_power": prefix + (degrees,),
              "cross_power": prefix + (degrees,), "truth_power": (leads, channels, degrees),
              "delta_grid_cross": (leads, 2, 11, channels), "delta_grid_norm": (leads, 2, 11, channels)}
    shapes.update({key: (leads, 2, 11, channels, degrees) for key in
                   ("delta_power", "forecast_delta_cross", "delta_truth_cross")})
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"Unexpected {key} layout: {arrays[key].shape} != {shape}")
    if metadata["base_names"] != BASE_NAMES or metadata["branches"] != ["GC", manifest["model_id"]]:
        raise ValueError("Unexpected smoother or predictor order")
    if (metadata["schema"] != "v24_smoothing_sweep_statistics_v1" or
            metadata["base_width_degrees"] != [*WIDTHS, None, None]):
        raise ValueError("Unexpected smoothing moment schema or widths")


def json_finite(value):
    """Undefined ratios are JSON null; retain their counts and raw moments."""
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_finite(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_parent_endpoints(manifest, initialization_ids, candidate_losses):
    """Repeated forecast/end-point differences are advisory, with provenance."""
    from scripts.experiments.run_v24_mgda import check, pinned, unseal
    if "parent_campaign" not in manifest:
        return None
    check(manifest["parent_campaign"])
    parent_root = Path(manifest["parent_campaign"]["path"]).parent
    parent_report = unseal(parent_root / "report.json")
    parent_manifest = unseal(parent_root / "manifest.json")
    if parent_report["campaign_hash"] != parent_manifest["record_hash"] or parent_report["status"] != "complete":
        raise ValueError("Parent smoothing report is incomplete or mismatched")
    check(parent_report["pooled_statistics"])
    with np.load(parent_report["pooled_statistics"]["path"], allow_pickle=False) as archive:
        parent_metadata = json.loads(str(archive["metadata"]))
        reference = archive["exact_gc_loss"]
    if (parent_metadata["initialization_ids"] != initialization_ids or
            parent_metadata["branches"] != ["GC", manifest["model_id"]] or
            parent_metadata["variants"] != ["raw", "gaussian", "band"]):
        raise ValueError("Parent repeated-endpoint comparison has a different cohort/layout")
    observed = candidate_losses[..., [0, 19, 29]]
    if observed.shape != reference.shape or not np.isfinite(reference).all():
        raise ValueError("Invalid parent exact-loss statistics")
    difference = observed - reference
    bound = 1e-5 * np.maximum(np.abs(reference), np.finfo(float).tiny)
    return {"parent_report": pinned(parent_root / "report.json"),
            "parent_pooled_statistics": parent_report["pooled_statistics"],
            "endpoints": ["raw", "fitted_gaussian_alpha1", "fitted_band_alpha1"],
            "parent_rollout_loss": reference.mean(axis=0).tolist(),
            "current_rollout_loss": observed.mean(axis=0).tolist(),
            "difference_by_lead": difference.tolist(), "reference_bound_by_lead": bound.tolist(),
            "relative_reference": 1e-5, "max_absolute_difference": float(np.abs(difference).max()),
            "max_reference_bound_ratio": float(np.max(np.abs(difference) / bound)),
            "exceeds_reference": bool(np.any(np.abs(difference) > bound)),
            "policy": "Finite repeat differences are retained warnings; this is not an acceptance gate"}


def plot_results(image_dir, rows, power_curves, model_id, count):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    models = ("GC", model_id)
    series = ("fixed_gaussian", "fitted_gaussian", "fitted_band")
    labels = ("Fixed Gaussian width", "Fitted Gaussian blend", "Fitted band blend")
    styles = ("-", "--", ":")

    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(image_dir / f"{name}.{suffix}", dpi=190, bbox_inches="tight")
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for model_index, model in enumerate(models):
        raw = next(row for row in rows if row["model"] == model and row["series"] == "raw")
        color = ("#2471A3", "#D55E00")[model_index]
        for kind, label, style in zip(series, labels, styles):
            curve = [raw] + [row for row in rows if row["model"] == model and row["series"] == kind]
            axis.plot([r["rollout_mean_relative_power_error_pct"] for r in curve],
                      [r["exact_gc_rollout_loss"] for r in curve], color=color,
                      linestyle=style, marker="o", markersize=3.5, label=f"{model}: {label}")
        axis.scatter(raw["rollout_mean_relative_power_error_pct"], raw["exact_gc_rollout_loss"],
                     marker="X", color=color, s=90, zorder=5, label=f"{model}: raw")
        axis.annotate(f"  {model} raw", (raw["rollout_mean_relative_power_error_pct"],
                                       raw["exact_gc_rollout_loss"]), color=color, fontsize=9)
    axis.set_xlabel("Mean absolute relative power error (%) — lower is better")
    axis.set_ylabel("Exact GraphCast rollout loss — lower is better")
    axis.set_title(f"Output smoothing: loss / spectrum tradeoff\n{count} paired 2023 starts; 40 six-hour leads")
    axis.grid(alpha=.2)
    axis.legend(fontsize=8, loc="best", frameon=False)
    save(fig, "mse_spectrum_tradeoff")

    degrees = np.arange(1, power_curves.shape[-1] + 1)
    configurations = candidates()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True, constrained_layout=True)
    for column, (kind, label) in enumerate(zip(series, labels)):
        norm = Normalize(0, 8 if kind == "fixed_gaussian" else 1)
        for model_index, model in enumerate(models):
            axis = axes[model_index, column]
            axis.plot(degrees, power_curves[model_index, 0], color="black", linewidth=2, label="Raw")
            for candidate_index, configuration in enumerate(configurations):
                if configuration["series"] != kind:
                    continue
                strength = configuration["width_degrees"] if kind == "fixed_gaussian" else configuration["alpha"]
                axis.plot(degrees, power_curves[model_index, candidate_index],
                          color=plt.cm.viridis(norm(strength)), linewidth=1.1)
            axis.axhline(1, color="grey", linestyle="-.", linewidth=1, label="ERA5 = 1")
            axis.set_title(f"{model}: {label}")
            axis.grid(alpha=.18)
            axis.set_xlim(degrees[0], degrees[-1])
            axis.set_xlabel("Spherical harmonic degree ℓ")
            if column == 0:
                axis.set_ylabel("Mean forecast / ERA5 power\n(equal channel weights)")
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes[:, column],
                     label="Gaussian width (degrees)" if kind == "fixed_gaussian" else "Blend strength α",
                     fraction=.035, pad=.025)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Day-10 power spectra across smoothing strengths: {count} paired starts\n"
                 "Pool physical powers across starts before channelwise ratios; degree zero excluded", fontsize=13)
    save(fig, "day10_power_ratio_series")


def report(root, manifest=None, *, name=None):
    from scripts.experiments.run_v24_mgda import pinned, seal, unseal, write
    from scripts.experiments.run_v24_smoothing import load_statistics, statistics_paths
    from src.analysis.smoothing_observer import atomic_npz
    from src.analysis.spherical_smoothing import json_digest

    root = Path(root).resolve()
    manifest = unseal(root / "manifest.json") if manifest is None else manifest
    frozen = unseal(root / "filters.json")
    filter_digest = json_digest(frozen)
    paths = statistics_paths(root, manifest, "evaluate")
    if not paths:
        raise ValueError("No complete evaluation statistics")
    workspace = Path(manifest["workspace"])
    name = name or root.name
    data_dir = workspace / "plots/analyze_models/data/resolution_eval" / name
    image_dir = workspace / "plots/analyze_models/images/resolution_eval" / name
    data_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    aggregate, individual = None, []
    configurations = candidates()
    mean_drift, raw_loss_drift = 0., 0.
    for path in paths:
        arrays, metadata = load_statistics(path, manifest, "evaluate")
        validate_shapes(arrays, metadata, manifest)
        if metadata["filter_digest"] != filter_digest or metadata["layout"] != frozen["layout"]:
            raise ValueError("Statistics use a different frozen filter/layout")
        for key, expected in (("source_digest", manifest["source_digest"]),
                              ("checkpoint_sha256", manifest["checkpoint"]["sha256"])):
            if metadata[key] != expected:
                raise ValueError(f"Statistic {key} differs from campaign")
        if aggregate is None:
            aggregate = {key: np.zeros_like(arrays[key], dtype=np.float64) for key in MOMENT_KEYS}
        for key in MOMENT_KEYS:
            aggregate[key] += arrays[key]
        sample_losses = np.stack([exact_loss(reconstruct_grid(arrays, c["base"], c["alpha"]), frozen["layout"])
                                  for c in configurations], axis=-1)
        individual.append(sample_losses)
        if "mean_drift" in arrays:
            mean_drift = max(mean_drift, float(np.max(np.abs(arrays["mean_drift"]))))
        if "exact_gc_loss" in arrays:
            reference = arrays["exact_gc_loss"][:, :, 0]
            raw_loss_drift = max(raw_loss_drift, float(np.max(np.abs(sample_losses[:, :, 0] - reference))))
    # Average raw moments first. Applying absolute value before this pooling
    # would measure a different quantity and distort the amplitude tradeoff.
    pooled = {key: value / len(paths) for key, value in aggregate.items()}
    del aggregate, arrays
    if not all(np.isfinite(value).all() for value in pooled.values()):
        raise ValueError("Nonfinite pooled statistics")
    individual = np.stack(individual)
    layout = frozen["layout"]
    raw_gc_loss = float(exact_loss(pooled["grid_mse"], layout)[:, 0].mean())
    rows, degree_rows, loss_by_lead, macro_by_lead, power_curves = [], [], [], [], []
    correlation_by_lead, error_power_by_lead = [], []
    zero_truth = int(np.sum(pooled["truth_power"][..., 1:] == 0))
    negative_power, negative_mse, negative_error_power, correlation_excess = 0, 0, 0, 0.
    for candidate_index, configuration in enumerate(configurations):
        mse, prediction, cross = reconstruct(pooled, configuration["base"], configuration["alpha"])
        loss = exact_loss(mse, layout)
        if not all(np.isfinite(value).all() for value in (mse, prediction, cross, loss)):
            raise ValueError("Nonfinite reconstructed statistics")
        power_ratio = ratio(prediction, pooled["truth_power"][:, None])
        error_power = prediction + pooled["truth_power"][:, None] - 2 * cross
        relative_error_power = ratio(error_power, pooled["truth_power"][:, None])[..., 1:]
        mismatch = np.abs(power_ratio[..., 1:] - 1)
        macro = finite_mean(mismatch, axis=(-2, -1))
        with np.errstate(invalid="ignore"):
            correlation = ratio(cross, np.sqrt(prediction * pooled["truth_power"][:, None]))
        finite_correlations = np.abs(correlation[np.isfinite(correlation)])
        if finite_correlations.size:
            correlation_excess = max(correlation_excess, float(finite_correlations.max() - 1))
        negative_power += int(np.sum(prediction < 0))
        negative_mse += int(np.sum(mse < 0))
        negative_error_power += int(np.sum(error_power < 0))
        day10_curve = finite_mean(power_ratio[-1, ..., 1:], axis=-2)
        power_curves.append(day10_curve)
        loss_by_lead.append(loss)
        macro_by_lead.append(macro)
        correlation_by_lead.append(finite_mean(correlation[..., 1:], axis=(-2, -1)))
        error_power_by_lead.append(finite_mean(relative_error_power, axis=(-2, -1)))
        for model_index, model in enumerate(("GC", manifest["model_id"])):
            rollout = float(loss[:, model_index].mean())
            rows.append({"model": model, "candidate_index": candidate_index, **configuration,
                         "exact_gc_rollout_loss": rollout,
                         "rollout_loss_reduction_pct": float(100 * (1 - ratio(rollout, raw_gc_loss))),
                         "exact_gc_day10_loss": float(loss[-1, model_index]),
                         "rollout_mean_relative_power_error_pct": float(100 * finite_mean(mismatch[:, model_index])),
                         "day10_relative_power_error_pct": float(100 * macro[-1, model_index]),
                         "rollout_mean_relative_error_power": float(finite_mean(relative_error_power[:, model_index])),
                         "day10_mean_relative_error_power": float(finite_mean(relative_error_power[-1, model_index])),
                         "rollout_mean_spectral_correlation": float(finite_mean(correlation[:, model_index, :, 1:])),
                         "day10_mean_spectral_correlation": float(finite_mean(correlation[-1, model_index, :, 1:])),
                         "valid_rollout_power_bins": int(np.isfinite(mismatch[:, model_index]).sum()),
                         "undefined_rollout_power_bins": int((~np.isfinite(mismatch[:, model_index])).sum()),
                         "undefined_rollout_correlation_bins": int((~np.isfinite(correlation[:, model_index, :, 1:])).sum())})
            for degree, value in enumerate(day10_curve[model_index], start=1):
                degree_rows.append({"model": model, "candidate": configuration["name"],
                                    "lead_days": manifest["target_steps"] / 4, "degree": degree,
                                    "mean_channel_power_ratio": float(value),
                                    "mean_channel_absolute_relative_power_error": float(finite_mean(
                                        mismatch[-1, model_index, :, degree - 1]))})
    candidate_losses = np.stack(loss_by_lead, axis=-1)
    repeated = compare_parent_endpoints(manifest, [path.stem for path in paths], candidate_losses)
    warnings = []
    if repeated and repeated["exceeds_reference"]:
        warnings.append("Repeated raw/fitted endpoint loss differs from parent by more than 1e-5 relative; values retained")
    if mean_drift > 1e-8:
        warnings.append("Finite mean drift exceeds 1e-8 physical units; values retained")
    if raw_loss_drift > 1e-10:
        warnings.append("Finite reconstructed raw-loss drift exceeds 1e-10; values retained")
    if negative_power or negative_mse or negative_error_power:
        warnings.append("Finite negative polynomial moments retained without clipping; inspect cancellation diagnostics")
    if correlation_excess > 3e-4:
        warnings.append("Finite spectral correlation exceeds magnitude 1+3e-4; values retained")
    diagnostics = {"warnings": warnings, "mean_drift_max_absolute": mean_drift,
                   "mean_drift_reference_absolute": 1e-8, "raw_loss_reconstruction_max_absolute": raw_loss_drift,
                   "raw_loss_reconstruction_reference_absolute": 1e-10,
                   "negative_reconstructed_power_bins": negative_power, "negative_reconstructed_grid_mse_bins": negative_mse,
                   "negative_reconstructed_error_power_bins": negative_error_power,
                   "correlation_max_excess_above_one": correlation_excess, "correlation_reference_excess": 3e-4,
                   "zero_truth_positive_degree_bins": zero_truth, "repeated_endpoint_comparison": repeated}
    pooled_path = root / "annual_pooled.npz"
    atomic_npz(pooled_path, {**pooled, "exact_gc_loss": candidate_losses,
        "macro_power_mismatch": np.stack(macro_by_lead, axis=-1),
        "macro_spectral_correlation": np.stack(correlation_by_lead, axis=-1),
        "macro_relative_error_power": np.stack(error_power_by_lead, axis=-1),
        "per_initialization_exact_gc_loss": individual},
        {"campaign_hash": manifest["record_hash"], "filter_digest": filter_digest,
         "initialization_ids": [path.stem for path in paths], "branches": ["GC", manifest["model_id"]],
         "base_names": BASE_NAMES, "candidates": configurations, "layout": layout,
         "ratio_policy": "pool physical moments across starts, then ratios; zero truth remains NaN",
         "refinement": "Any scalar blend strength can be reconstructed offline from pooled displacement moments"})
    write_csv(data_dir / "summary.csv", rows)
    write_csv(data_dir / "day10_power_ratio_by_degree.csv", degree_rows)
    plot_results(image_dir, rows, np.stack(power_curves, axis=1), manifest["model_id"], len(paths))
    result = {"status": "complete", "count": len(paths), "campaign_hash": manifest["record_hash"],
              "filter": pinned(root / "filters.json"), "pooled_statistics": pinned(pooled_path),
              "script": pinned(__file__), "numerical_diagnostics": diagnostics,
              "branches": ["GC", manifest["model_id"]], "candidates": configurations,
              "summary": rows, "scientific_improvement_required": False, "no_clipping": True,
              "metric": "Exact native-grid GraphCast loss; power error is mean absolute pooled forecast/truth ratio minus one, equally across channels, positive degrees and leads",
              "uncertainty": "Fixed-checkpoint descriptive estimates on reused 2023 cohort; no confidence intervals or training-seed claims",
              "selection": "Predeclared strengths; no fitting or selecting a winning smoother on 2023",
              "data_dir": str(data_dir), "image_dir": str(image_dir)}
    result["artifacts"] = [pinned(path) for path in sorted(data_dir.glob("*.csv"))] + [
        pinned(path) for path in sorted(image_dir.glob("*")) if path.suffix in (".png", ".pdf")]
    result = seal(root / "report.json", json_finite(result))
    write(data_dir / "summary.json", result)
    relative_images = f"../../../images/resolution_eval/{name}"
    table = ["| Model | Smoother | Exact GC loss | Loss reduction (%) | Mean power error (%) |",
             "| --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        table.append(f"| {row['model']} | {row['name']} | {row['exact_gc_rollout_loss']:.5f} | "
                     f"{row['rollout_loss_reduction_pct']:.3f} | {row['rollout_mean_relative_power_error_pct']:.3f} |")
    (data_dir / "report.md").write_text("\n".join([
        "# GC/M1 smoothing strength tradeoffs", "",
        f"{len(paths)} paired 2023 initializations, {manifest['target_steps']} six-hour leads, fixed GC and M1 checkpoints.", "",
        "The three series are global Gaussian widths (0–8 degrees), independently development-fitted Gaussian profiles blended with the raw forecast, and independently development-fitted band profiles blended with the raw forecast. Blend strengths are 0, 0.1, …, 1. All zero strengths share the raw forecast. Filters use the original 2022 development fit; these curves do not refit or select a smoother on 2023. Filtered outputs are scored without feeding them into subsequent forecast steps.", "",
        f"![Loss and power tradeoff]({relative_images}/mse_spectrum_tradeoff.png)", "",
        "Loss is the exact native-grid GraphCast objective, reconstructed from physical grid-error cross products with its original latitude, variable, pressure and difference-scale weights. Loss reductions use pooled raw GC loss as their common denominator. Mean power error pools physical spherical powers across starts first, then averages 100×|forecast/truth power−1| equally over channels, positive degrees and leads. It measures amplitude fidelity; it is distinct from spatial alignment and the weighted GC loss.", "",
        f"![Day-10 spectrum series]({relative_images}/day10_power_ratio_series.png)", "",
        "The spectrum curves average dimensionless channelwise ratios, allowing excesses and deficits to offset. The absolute-error metric in the tradeoff figure does not allow that cancellation. Degree zero is excluded; no climatology subtraction or positivity clipping is applied.", "",
        "summary.csv also gives mean spectral correlation and relative forecast-error power, averaged over the same positive degrees and channels. Error power is forecast power + truth power − 2×cross power, divided by pooled truth power. These diagnostics preserve alignment information alongside forecast amplitude. Undefined ratios are excluded with explicit counts.", "",
        *table, "", "Results describe fixed checkpoints on the reused 2023 cohort. No confidence intervals are estimated here. Finite numerical drift and cancellation warnings are retained in summary.json and do not block completion. Nonfinite or mismatched source statistics are errors. The pooled displacement moments allow additional blend strengths to be evaluated offline without new forecasts.", ""]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--name")
    args = parser.parse_args()
    result = report(args.campaign, name=args.name)
    print(json.dumps({key: result[key] for key in ("status", "count", "data_dir", "image_dir")}, indent=2))


if __name__ == "__main__":
    main()
