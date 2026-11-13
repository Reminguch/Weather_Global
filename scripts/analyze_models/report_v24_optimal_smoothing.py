#!/usr/bin/env python3
"""Report frozen 2022-fitted spherical filters on the v4 2023 cohort.

Native-grid GraphCast loss and continuous spectral MSE are reported separately.
The historical v4 GC trajectory is the canonical native-loss denominator;
new unfiltered trajectories also remain visible as repeatability measurements.
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

VARIANTS = ("raw", "projected", "optimal")
BOOTSTRAP = {"replicates": 10000, "block_length": 5, "seed": 20260916}
DRIFT_REFERENCE = 1e-5


def ratio(numerator, denominator):
    """Zero denominators remain undefined, without a numerical floor."""
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    output = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        np.divide(numerator, denominator, out=output, where=denominator > 0)
    if np.isinf(output).any():
        raise ValueError("Nonfinite ratio with positive denominator")
    return output


def json_finite(value):
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_finite(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_finite(row) for row in rows)


def load_v4_reference(manifest, initialization_ids):
    """Align pinned historical raw GC/M1 losses by date, never shard position."""
    from scripts.experiments.run_v24_mgda import check

    records = {}
    expected_leads = list(range(6, 6 * manifest["target_steps"] + 1, 6))
    baseline_hash = manifest["baseline"].get("sha256", manifest["baseline"].get("checkpoint_sha256"))
    for pin in manifest["v4_reference_metrics"]:
        check(pin)
        payload = json.loads(Path(pin["path"]).read_text())
        if (payload["model_id"] != manifest["model_id"] or
                payload["checkpoint_sha256"] != manifest["checkpoint"]["sha256"] or
                payload["baseline_checkpoint_sha256"] != baseline_hash):
            raise ValueError("Canonical v4 checkpoint identity differs")
        for record in payload["original_graphcast_loss_per_initialization"]:
            identity = record["initialization_id"]
            if identity in records or record["lead_hours"] != expected_leads:
                raise ValueError("Duplicated v4 date or mismatched lead schedule")
            value = np.stack((record["baseline_per_step"], record["full_per_step"]), axis=-1)
            if value.shape != (manifest["target_steps"], 2) or not np.isfinite(value).all():
                raise ValueError("Incomplete/nonfinite canonical v4 losses")
            records[identity] = value
    if set(records) != set(initialization_ids) or len(initialization_ids) != len(records):
        raise ValueError("Canonical v4 and new smoothing cohorts differ")
    return np.stack([records[identity] for identity in initialization_ids])


def validate_shapes(arrays, metadata, manifest, layout):
    leads, channels, degrees = manifest["target_steps"], len(layout["channels"]), manifest["policy"]["lmax"] + 1
    shapes = {"exact_gc_loss": (leads, 2, 3), "grid_mse": (leads, 2, 3, channels),
              "spectral_weighted_gc_loss": (leads, 2, 3),
              "truth_power": (leads, channels, degrees),
              "forecast_projection_residual_mse": (leads, 2, channels),
              "truth_projection_residual_mse": (leads, channels),
              "mean_drift": (leads, 2, 3, channels),
              "prediction_mean": (leads, 2, 3, channels),
              "truth_mean": (leads, channels)}
    for key in ("prediction_power", "cross_power", "error_power"):
        shapes[key] = (leads, 2, 3, channels, degrees)
    for key, shape in shapes.items():
        if key not in arrays or arrays[key].shape != shape:
            raise ValueError(f"Unexpected or missing {key} shape; expected {shape}")
    for key in ("negative_fraction", "positive_field_minimum"):
        if key not in arrays or arrays[key].ndim != 4 or arrays[key].shape[:3] != (leads, 2, 3):
            raise ValueError(f"Unexpected or missing {key} shape")
    if metadata["branches"] != ["GC", manifest["model_id"]] or tuple(metadata["variants"]) != VARIANTS:
        raise ValueError("Predictor or filter variant order changed")


def bootstrap_endpoints(individual, canonical, *, replicates=10000, block_length=5, seed=20260916):
    """Paired non-wrapping date blocks, pooling losses before every ratio."""
    count = len(individual)
    if not 1 <= block_length <= count or replicates < 1:
        raise ValueError("Invalid paired bootstrap dimensions")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, count - block_length + 1,
                          size=(replicates, (count + block_length - 1) // block_length))
    indices = (starts[..., None] + np.arange(block_length)).reshape(replicates, -1)[:, :count]
    weights = np.zeros((replicates, count), dtype=np.float64)
    np.add.at(weights, (np.arange(replicates)[:, None], indices), 1. / count)
    current_endpoints = np.stack((individual.mean(axis=1), individual[:, -1]), axis=1)
    reference_endpoints = np.stack((canonical[:, :, 0].mean(axis=1), canonical[:, -1, 0]), axis=1)
    sampled = (weights @ current_endpoints.reshape(count, -1)).reshape(replicates, 2, 2, 3)
    reference = weights @ reference_endpoints
    improvements = 100 * (1 - ratio(sampled, reference[:, :, None, None]))
    pair = 100 * (1 - ratio(sampled[:, :, 1, 2], sampled[:, :, 0, 2]))
    def interval(values):
        if np.isnan(values).any():
            # Do not silently remove undefined bootstrap draws.
            return np.full((2,) + values.shape[1:], np.nan)
        return np.percentile(values, [2.5, 97.5], axis=0)
    return {"canonical_improvement_ci95": interval(improvements),
            "filtered_pair_reduction_ci95": interval(pair),
            "replicates": replicates, "block_length": block_length, "seed": seed,
            "endpoints": ["rollout", "day10"],
            "interpretation": "Pointwise paired forecast-sample intervals for fixed checkpoints/filters; no training-seed or between-year uncertainty"}


def drift_diagnostics(individual, canonical):
    observed = individual[..., 0]
    delta = observed - canonical
    bound = DRIFT_REFERENCE * np.abs(canonical)
    return {"branches": ["GC", "M1"], "reference_relative": DRIFT_REFERENCE,
            "reference_absolute": 0., "canonical_rollout": canonical.mean(axis=(0, 1)),
            "current_raw_rollout": observed.mean(axis=(0, 1)),
            "rollout_relative_difference_pct": 100 * ratio(delta.mean(axis=(0, 1)), canonical.mean(axis=(0, 1))),
            "maximum_absolute_difference": np.abs(delta).max(axis=(0, 1)),
            "maximum_absolute_relative_difference_pct": 100 * np.nanmax(ratio(np.abs(delta), canonical), axis=(0, 1)),
            "exceeds_reference": bool(np.any(np.abs(delta) > bound)),
            "policy": "Finite repeated-run differences are warnings, not acceptance criteria"}


def plot_summary(root, pooled, canonical_by_lead, power_mismatch, model_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    days = np.arange(1, pooled["exact_gc_loss"].shape[0] + 1) / 4
    for branch, name in enumerate(("GC", model_id)):
        for variant, label in enumerate(VARIANTS):
            loss = pooled["exact_gc_loss"][:, branch, variant]
            ys = (loss, 100 * (1 - ratio(loss, canonical_by_lead)),
                  pooled["spectral_weighted_gc_loss"][:, branch, variant],
                  100 * power_mismatch[:, branch, variant])
            for axis, values in zip(axes.flat, ys):
                axis.plot(days, values, color=f"C{branch}", linestyle=("-", ":", "--")[variant],
                          label=f"{name} {label}")
    axes[0, 0].plot(days, canonical_by_lead, color="0.45", lw=1, label="v4 canonical GC")
    for axis, label in zip(axes.flat, ("Exact native-grid GC loss", "Loss reduction vs v4 GC (%)",
                                      "GC-weighted spectral MSE\n(distinct from native-grid loss)",
                                      "Mean relative power mismatch (%)\n(equal channel / positive-degree weights)")):
        axis.set(xlabel="Forecast lead (days)", ylabel=label)
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(root / f"summary.{suffix}", dpi=180)
    plt.close(figure)


def report(root, manifest=None):
    from scripts.experiments import run_v24_smoothing as original
    from scripts.experiments.run_v24_mgda import check, pinned, seal, unseal
    from src.analysis.smoothing_observer import atomic_npz
    from src.analysis.spherical_smoothing import json_digest

    root = Path(root).resolve()
    manifest = unseal(root / "manifest.json") if manifest is None else manifest
    if (manifest["format"] != "v24_optimal_smoothing_campaign_v1" or manifest["model_id"] != "M1" or
            manifest["policy"]["fit_year"] != 2022 or manifest["policy"]["evaluation_year"] != 2023):
        raise ValueError("Unexpected optimal-smoothing campaign scope")
    check(manifest["manuscript_reference"])
    frozen = unseal(root / "filters.json")
    if frozen["campaign_hash"] != manifest["record_hash"]:
        raise ValueError("Filters belong to a different campaign")
    paths = original.statistics_paths(root, manifest, "evaluate")
    if not paths:
        raise ValueError("No completed annual statistics")
    filter_digest = json_digest(frozen)
    layout = frozen["layout"]
    factor = np.asarray(layout["weights"], dtype=np.float64) / np.asarray(layout["scales"], dtype=np.float64)**2
    if not np.isfinite(factor).all():
        raise ValueError("Nonfinite channel normalization")
    aggregate, minimum, individual = None, None, []
    exact_consistency, spectral_consistency, max_mean_drift = 0., 0., 0.
    for path in paths:
        arrays, metadata = original.load_statistics(path, manifest, "evaluate")
        validate_shapes(arrays, metadata, manifest, layout)
        if metadata["filter_digest"] != filter_digest or metadata["layout"] != layout:
            raise ValueError("Annual filter/layout differs from frozen development fit")
        if (metadata["source_digest"] != manifest["source_digest"] or
                metadata["checkpoint_sha256"] != manifest["checkpoint"]["sha256"]):
            raise ValueError("Annual source/checkpoint identity changed")
        if aggregate is None:
            aggregate = {key: np.zeros_like(value, dtype=np.float64) for key, value in arrays.items()}
            minimum = arrays["positive_field_minimum"].copy()
        if set(arrays) != set(aggregate):
            raise ValueError("Annual statistic keys changed")
        for key, value in arrays.items():
            aggregate[key] += value
        minimum = np.minimum(minimum, arrays["positive_field_minimum"])
        individual.append(arrays["exact_gc_loss"])
        exact_consistency = max(exact_consistency, float(np.max(np.abs(
            np.sum(arrays["grid_mse"] * factor, axis=-1) - arrays["exact_gc_loss"]))))
        spectral_consistency = max(spectral_consistency, float(np.max(np.abs(
            np.sum(arrays["error_power"].sum(axis=-1) * factor, axis=-1) - arrays["spectral_weighted_gc_loss"]))))
        max_mean_drift = max(max_mean_drift, float(np.max(np.abs(arrays["mean_drift"]))))
    pooled = {key: value / len(paths) for key, value in aggregate.items()}
    if not all(np.isfinite(value).all() for value in pooled.values()):
        raise ValueError("Nonfinite pooled raw statistics")
    individual = np.stack(individual)
    ids = [path.stem for path in paths]
    canonical = load_v4_reference(manifest, ids)
    drift = drift_diagnostics(individual, canonical)
    canonical_by_lead = canonical[:, :, 0].mean(axis=0)
    canonical_rollout = canonical_by_lead.mean()
    power_ratio = ratio(pooled["prediction_power"], pooled["truth_power"][:, None, None])
    mismatch = np.abs(power_ratio[..., 1:] - 1)
    valid = np.isfinite(mismatch)
    macro = ratio(np.where(valid, mismatch, 0.).sum(axis=(-2, -1)), valid.sum(axis=(-2, -1)))
    physical_mse = pooled["error_power"].sum(axis=-1)
    physical_rmse = np.sqrt(np.where(physical_mse >= 0, physical_mse, np.nan))
    bootstrap = bootstrap_endpoints(individual, canonical, **BOOTSTRAP)
    summaries, perlead, channels, perdate = [], [], [], []
    for bi, branch in enumerate(("GC", manifest["model_id"])):
        for vi, variant in enumerate(VARIANTS):
            losses = pooled["exact_gc_loss"][:, bi, vi]
            row = {"model": branch, "variant": variant, "count": len(paths),
                   "exact_gc_rollout_loss": float(losses.mean()), "exact_gc_day10_loss": float(losses[-1]),
                   "rollout_improvement_pct": float(100 * (1 - ratio(losses.mean(), canonical_rollout))),
                   "day10_improvement_pct": float(100 * (1 - ratio(losses[-1], canonical_by_lead[-1]))),
                   "rollout_vs_own_raw_reduction_pct": float(100 * (1 - ratio(losses.mean(), pooled["exact_gc_loss"][:, bi, 0].mean()))),
                   "gc_weighted_spectral_rollout_mse": float(pooled["spectral_weighted_gc_loss"][:, bi, vi].mean()),
                   "rollout_mean_relative_power_error_pct": float(100 * np.mean(macro[:, bi, vi]))}
            for ei, endpoint in enumerate(("rollout", "day10")):
                for ci, name in enumerate(("lower", "upper")):
                    row[f"{endpoint}_improvement_ci95_{name}"] = bootstrap["canonical_improvement_ci95"][ci, ei, bi, vi]
            summaries.append(row)
            for li, loss in enumerate(losses):
                perlead.append({"model": branch, "variant": variant, "lead_hours": 6 * (li + 1),
                    "exact_gc_loss": loss, "canonical_v4_gc_loss": canonical_by_lead[li],
                    "improvement_pct": 100 * (1 - ratio(loss, canonical_by_lead[li])),
                    "gc_weighted_spectral_mse": pooled["spectral_weighted_gc_loss"][li, bi, vi],
                    "mean_relative_power_error_pct": 100 * macro[li, bi, vi]})
            for ci, (variable, level) in enumerate(layout["channels"]):
                values = physical_mse[:, bi, vi, ci]
                for lead, mse in [("rollout", values.mean()), *[(6 * (li + 1), v) for li, v in enumerate(values)]]:
                    channels.append({"model": branch, "variant": variant, "variable": variable,
                        "pressure_level": level, "lead_hours": lead, "physical_spectral_mse": mse,
                        "physical_spectral_rmse": np.sqrt(mse) if mse >= 0 else np.nan})
            for ni, identity in enumerate(ids):
                for li in range(manifest["target_steps"]):
                    perdate.append({"initialization_id": identity, "lead_hours": 6 * (li + 1),
                        "model": branch, "variant": variant,
                        "exact_gc_loss": individual[ni, li, bi, vi],
                        "canonical_v4_gc_loss": canonical[ni, li, 0],
                        "canonical_v4_m1_loss": canonical[ni, li, 1]})
    own_optimal = pooled["exact_gc_loss"][:, :, 2]
    filtered_pair = {"definition": "100*(1-pooled M1 optimal native-grid loss / pooled GC optimal native-grid loss)",
                    "rollout_reduction_pct": 100 * (1 - ratio(own_optimal[:, 1].mean(), own_optimal[:, 0].mean())),
                    "day10_reduction_pct": 100 * (1 - ratio(own_optimal[-1, 1], own_optimal[-1, 0])),
                    "ci95": bootstrap["filtered_pair_reduction_ci95"]}
    warnings = []
    if drift["exceeds_reference"]:
        warnings.append("Finite raw-forecast differences from v4 exceed the 1e-5 relative reference; all evaluations retained")
    if exact_consistency > DRIFT_REFERENCE or spectral_consistency > DRIFT_REFERENCE:
        warnings.append("Finite saved-versus-recomputed loss differences exceed the 1e-5 absolute reference; values retained")
    if np.any(physical_mse < 0):
        warnings.append("Finite negative summed spectral error power retained; its RMSE is undefined")
    diagnostics = {"repeated_v4_raw": drift, "exact_loss_reducer_max_absolute_difference": exact_consistency,
                   "spectral_loss_reducer_max_absolute_difference": spectral_consistency,
                   "reducer_reference_absolute": DRIFT_REFERENCE, "mean_drift_max_absolute": max_mean_drift,
                   "mean_policy": "Direct spherical reconstruction does not impose native-grid mean preservation",
                   "zero_truth_power_bins": int(np.count_nonzero(pooled["truth_power"] == 0)),
                   "valid_power_ratio_bins_by_lead": valid.sum(axis=(-2, -1)), "warnings": warnings}
    archive = {**pooled, "positive_field_minimum_over_starts": minimum,
               "power_ratio": power_ratio, "macro_power_mismatch": macro,
               "physical_channel_spectral_mse": physical_mse,
               "physical_channel_spectral_rmse": physical_rmse,
               "per_initialization_exact_gc_loss": individual,
               "canonical_v4_raw_loss_per_initialization": canonical,
               "raw_repeat_difference_per_initialization": individual[..., 0] - canonical,
               "raw_repeat_reference_bound_per_initialization": DRIFT_REFERENCE * np.abs(canonical)}
    archive_metadata = {"campaign_hash": manifest["record_hash"], "filter_digest": filter_digest,
        "initialization_ids": ids, "branches": ["GC", manifest["model_id"]], "variants": list(VARIANTS),
        "layout": layout, "lead_hours": list(range(6, 6 * manifest["target_steps"] + 1, 6)),
        "physical_spectral_mse": "Sum error power over all represented degrees including degree zero; physical units squared per channel",
        "ratio_policy": "Pool raw moments across initialization dates before ratios; zero denominators undefined",
        "canonical_v4_reference_metrics": manifest["v4_reference_metrics"]}
    # A partially completed report can resume without mutating its statistics.
    pooled_path = root / "annual_pooled.npz"
    if pooled_path.exists():
        with np.load(pooled_path, allow_pickle=False) as existing:
            if json.loads(str(existing["metadata"])) != archive_metadata or set(existing.files) != {*archive, "metadata"}:
                raise ValueError("Existing report archive has different provenance/layout")
            if any(not np.array_equal(existing[key], value, equal_nan=True) for key, value in archive.items()):
                raise ValueError("Existing report archive has different statistics")
    else:
        atomic_npz(pooled_path, archive, archive_metadata)
    for name, rows in (("summary", summaries), ("perlead", perlead),
                       ("physical_channel_spectral", channels), ("per_initialization", perdate)):
        write_csv(root / f"{name}.csv", rows)
    plot_summary(root, pooled, canonical_by_lead, macro, manifest["model_id"])
    result = {"status": "complete", "count": len(paths), "campaign_hash": manifest["record_hash"],
              "filter": pinned(root / "filters.json"), "pooled_statistics": pinned(pooled_path),
              "manuscript_reference": manifest["manuscript_reference"],
              "canonical_v4_reference_metrics": manifest["v4_reference_metrics"],
              "branches": ["GC", manifest["model_id"]], "variants": list(VARIANTS),
              "native_loss_definition": "Exact original GraphCast native-grid objective; improvement=100*(1-pooled loss/pooled historical v4 GC loss)",
              "spectral_definition": "Physical channel MSE is sum of error power across represented degrees; GC-weighted spectral MSE also applies channel weights/difference normalization and is distinct from native-grid loss",
              "spherical_transform": "Raw/projected spectral diagnostics use Af; optimal uses fitted b_l*Af",
              "canonical_v4_gc_rollout_loss": canonical_rollout,
              "canonical_v4_gc_loss_by_lead": canonical_by_lead,
              "summary": summaries, "filtered_m1_vs_filtered_gc": filtered_pair,
              "bootstrap": bootstrap, "numerical_diagnostics": diagnostics,
              "scientific_improvement_required": False, "no_clipping": True,
              "selection": "Filters fitted only on 2022; 2023 evaluated without filter selection",
              "known_exposure": "2023 informed prior analyses; this is not an untouched test year",
              "data_outputs": {name: pinned(root / f"{name}.csv") for name in
                  ("summary", "perlead", "physical_channel_spectral", "per_initialization")}}
    result = json_finite(result)
    lines = ["# GraphCast optimal spherical filtering: v4 GC and M1", "",
             f"Filters fitted on 2022 forecasts; evaluated on {len(paths)} v4 initialization dates in 2023.", "",
             "Native-grid improvements use the original v4 GC trajectory as denominator. Losses are pooled before ratios. Projected fields isolate the change caused by spherical projection; optimal fields use the frozen degree-wise filter.", "",
             "| Model | Output | Rollout exact GC loss | Reduction vs v4 GC (%) | Day-10 reduction (%) |", "| --- | --- | ---: | ---: | ---: |"]
    for row in summaries:
        lines.append(f"| {row['model']} | {row['variant']} | {row['exact_gc_rollout_loss']:.8g} | {row['rollout_improvement_pct']:.4f} | {row['day10_improvement_pct']:.4f} |")
    lines += ["", "The independently filtered M1-versus-filtered-GC comparison uses the two new filtered trajectories; it is reported separately in `report.json`.", "",
              "`physical_channel_spectral.csv` contains per-channel physical spectral MSE/RMSE. `gc_weighted_spectral_mse` additionally uses GC normalization/weights; neither spectral quantity is the exact native-grid objective.", "",
              "`per_initialization.csv` and `annual_pooled.npz` retain paired date/lead data, historical reference losses, repeat differences, projection residuals, mean changes, and positivity diagnostics. Raw spherical diagnostics and projected diagnostics use the same coefficients; their native-grid losses can differ.", "",
              "Pointwise 95% intervals use 10,000 non-wrapping paired bootstrap draws with five-start blocks and seed 20260916. They describe forecast-sample uncertainty for fixed checkpoints/filters, not training-seed variation or simultaneous confidence bands. The 2023 cohort has prior analysis exposure.", "",
              "Finite numerical differences are retained as warnings; completing this evaluation does not require demonstrating improvement."]
    for warning in warnings:
        lines += ["", f"Warning: {warning}."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    result["readout"] = pinned(root / "REPORT.md")
    return seal(root / "report.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
