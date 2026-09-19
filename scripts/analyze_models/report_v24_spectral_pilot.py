"""Validate and plot the exploratory V24 res1 zonal Fourier pilot.

Run through ``scripts/graphcast_env.sh``. This consumes complete per-start NPZ
records and a frozen protocol; it does not run forecasts or accept partial runs.
Powers and cospectra are averaged over starts *before* constructing ratios or
correlations. Wavenumber is cycles around a latitude circle, not a single
physical wavelength. The initialization year and pilot scope come from the
frozen protocol. A small pilot provides no annual or no-harm conclusion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SPECTRAL_KEYS = (
    "truth_power", "baseline_power", "full_power", "baseline_cross_power",
    "full_cross_power", "baseline_error_power", "full_error_power",
)
REPEATED_KEYS = (
    "truth_power", "baseline_power", "baseline_cross_power", "baseline_error_power",
)
DISPLAY_LEADS = (24, 72, 168, 240)
MODEL_IDS = ("M1", "M2", "M3")
VARIABLE_UNITS = {
    "T2m": "K²", "U10": "m² s⁻²", "V10": "m² s⁻²", "MSLP": "Pa²",
    "TP6": "m²", "T850": "K²", "Q700": "(kg kg⁻¹)²", "U850": "m² s⁻²",
    "V850": "m² s⁻²", "WS10": "m² s⁻²", "WS850": "m² s⁻²",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _time(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, f"Initialization time lacks timezone: {value}")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _evaluation_year(protocol):
    # Legacy protocols predate this explicit field and contain 2022 cases.
    year = protocol.get("evaluation_year", 2022)
    _require(type(year) is int and 1 <= year <= 9999,
             "evaluation_year must be an integer calendar year")
    return year


def _divide(numerator, denominator):
    out = np.full(np.broadcast_shapes(np.shape(numerator), np.shape(denominator)), np.nan)
    return np.divide(numerator, denominator, out=out, where=np.asarray(denominator) > 0)


def _json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _validated_exact_loss(workflow, protocol):
    """Require the workflow's exact-loss checks and bind them to current files."""
    path = workflow / "exact_loss_validation.json"
    record, sha256 = _read_json(path)
    _require(record.get("status") == "passed", "Exact GraphCast loss validation has not passed")
    _require(record.get("protocol_hash") == protocol["protocol_hash"], "Exact-loss protocol_hash mismatch")
    _require(record.get("canonical_baseline_model_id") == "M1", "Exact-loss canonical baseline must be M1")
    _require([model["model_id"] for model in record.get("models", [])] == list(MODEL_IDS),
             "Exact-loss validation missing models")
    sources = record.get("evaluation_sources", [])
    expected = {(workflow / "models" / model / "evaluation.json").resolve() for model in MODEL_IDS}
    _require(len(sources) == len(expected) and {Path(source["path"]).resolve() for source in sources} == expected,
             "Exact-loss evaluation_sources mismatch")
    for source in sources:
        digest = hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest()
        _require(digest == source["sha256"], f"Exact-loss validation is stale: {source['path']}")
    return record, {"path": str(path), "sha256": sha256}


def _validate_spectral_identity(arrays, path):
    truth = arrays["truth_power"]
    for branch in ("baseline", "full"):
        power = arrays[f"{branch}_power"]
        cross = arrays[f"{branch}_cross_power"]
        error = arrays[f"{branch}_error_power"]
        bound = np.sqrt(truth) * np.sqrt(power)
        _require(np.all(np.abs(cross) <= bound * (1 + 1e-6) + 1e-24),
                 f"{path}: {branch} cospectrum violates Cauchy-Schwarz")
        # Compare on the power scale: subtracting almost equal spectra can
        # produce relative roundoff in tiny errors even when both are valid.
        tolerance = 1e-6 * (truth + power) + 1e-24
        _require(np.all(np.abs(error - (truth + power - 2 * cross)) <= tolerance),
                 f"{path}: {branch} spectral error identity failed")


def _load_record(path, protocol, model, expected_times, expected_k=None):
    raw = path.read_bytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as source:
        required = set(SPECTRAL_KEYS) | {"wavenumber", "lead_hours", "variables", "metadata"}
        _require(required.issubset(source.files), f"{path}: missing NPZ fields {required - set(source.files)}")
        metadata = json.loads(str(source["metadata"].item()))
        _require(metadata.get("status") == "complete", f"{path}: initialization status is not complete")
        for key, expected in (
            ("model_id", model["model_id"]), ("checkpoint_sha256", model["checkpoint_sha256"]),
            ("protocol_hash", protocol["protocol_hash"]),
            ("baseline_checkpoint_sha256", protocol["baseline"]["checkpoint_sha256"]),
        ):
            _require(metadata.get(key) == expected, f"{path}: {key} mismatch")
        _require(metadata.get("spectral_metadata") == protocol["spectral_metadata"],
                 f"{path}: spectral_metadata mismatch")
        initialization = _time(metadata["initialization_time"])
        _require(initialization in expected_times, f"{path}: unexpected initialization {initialization}")
        leads = np.asarray(source["lead_hours"])
        variables = source["variables"].astype(str).tolist()
        k = np.asarray(source["wavenumber"])
        _require(np.array_equal(leads, protocol["lead_hours"]), f"{path}: lead_hours mismatch")
        _require(variables == protocol["spectral_variables"], f"{path}: variables mismatch")
        _require(k.ndim == 1 and len(k) > 1 and np.array_equal(k, np.arange(len(k))),
                 f"{path}: wavenumber must contain consecutive integer modes including DC")
        if expected_k is not None:
            _require(np.array_equal(k, expected_k), f"{path}: wavenumber mismatch")
        shape = (len(leads), len(variables), len(k))
        arrays = {}
        for key in SPECTRAL_KEYS:
            value = np.asarray(source[key])
            _require(value.shape == shape, f"{path}: {key} shape {value.shape} != {shape}")
            _require(np.isrealobj(value) and np.isfinite(value).all(), f"{path}: {key} must be finite and real")
            _require(np.all(value[..., 0] == 0), f"{path}: {key} DC must be zero after zonal demeaning")
            if "cross" not in key:
                _require(np.all(value >= 0), f"{path}: {key} must be nonnegative")
            arrays[key] = value.astype(np.float64)
    _validate_spectral_identity(arrays, path)
    return initialization, k, arrays, {
        "path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
        "model_id": model["model_id"], "initialization_time": initialization,
    }


def load_spectra(workflow):
    """Load complete paired pilot records; reject provenance or baseline drift."""
    workflow = Path(workflow).resolve()
    protocol, protocol_file_sha256 = _read_json(workflow / "protocol.json")
    body = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _require(digest == protocol.get("protocol_hash"), "Frozen protocol_hash does not match protocol contents")
    expected_times = [_time(value) for value in protocol["initialization_times"]]
    _require(expected_times and len(set(expected_times)) == len(expected_times), "Duplicate/empty initialization_times")
    year = _evaluation_year(protocol)
    _require(all(datetime.fromisoformat(value.replace("Z", "+00:00")).year == year for value in expected_times),
             f"All initialization times must match evaluation_year {year}")
    models = protocol["models"]
    _require([model["model_id"] for model in models] == list(MODEL_IDS), "Expected ordered models M1, M2, M3")
    _require(protocol["baseline"]["canonical_model_id"] == "M1", "Canonical baseline must be M1")
    _require(len(protocol["lead_hours"]) == protocol["target_steps"], "target_steps/lead_hours mismatch")
    _require(protocol["spectral_variables"] and len(set(protocol["spectral_variables"])) == len(protocol["spectral_variables"]),
             "Duplicate/empty spectral_variables")
    _require(all(lead in protocol["lead_hours"] for lead in DISPLAY_LEADS), "Missing display leads 24/72/168/240 hours")
    rtol = protocol["baseline"].get("spectral_repeatability_rtol", 1e-3)
    atol = protocol["baseline"].get("spectral_repeatability_atol", 1e-10)
    _require(0 <= rtol <= 1e-3 and 0 <= atol <= 1e-10, "Spectral repeatability tolerances exceed pilot limits")
    source_audit, repeated_audit, pooled = [], {}, {}
    canonical, wavenumber = {}, None
    for model in models:
        model_id = model["model_id"]
        directory = workflow / "models" / model_id
        files = sorted(directory.glob("*.npz"))
        _require(len(files) == len(expected_times),
                 f"{model_id}: expected {len(expected_times)} initialization NPZs, found {len(files)}")
        records, sources = {}, {}
        for path in files:
            initialization, k, arrays, provenance = _load_record(path, protocol, model, expected_times, wavenumber)
            if wavenumber is None:
                wavenumber = k
            _require(initialization not in records, f"{model_id}: duplicate initialization {initialization}")
            records[initialization] = arrays
            sources[initialization] = provenance
        _require(set(records) == set(expected_times), f"{model_id}: missing initialization records")
        source_audit.extend(sources[time] for time in expected_times)
        if model_id == "M1":
            canonical = {time: {key: value for key, value in record.items() if key in REPEATED_KEYS}
                         for time, record in records.items()}
        else:
            checks = {key: {"max_absolute_difference": 0., "max_relative_difference": 0.} for key in REPEATED_KEYS}
            for time in expected_times:
                for key in REPEATED_KEYS:
                    value, reference = records[time][key], canonical[time][key]
                    difference = np.abs(value - reference)
                    check = checks[key]
                    check["max_absolute_difference"] = max(check["max_absolute_difference"], float(difference.max()))
                    nonzero = np.abs(reference) > 0
                    relative = float(np.max(difference[nonzero] / np.abs(reference[nonzero]))) if nonzero.any() else 0.
                    check["max_relative_difference"] = max(check["max_relative_difference"], relative)
                    matches = (np.array_equal(value, reference) if key == "truth_power"
                               else np.allclose(value, reference, rtol=rtol, atol=atol))
                    _require(matches,
                             f"{model_id} {time}: canonical M1 {key} repeatability failed "
                             f"(max abs={difference.max():.6g}, max relative={relative:.6g}; "
                             f"rtol={rtol}, atol={atol})")
            repeated_audit[model_id] = checks
        # All initializations receive equal weight. Each individual spectrum
        # already contains its normalized cosine-latitude weights.
        pooled[model_id] = {key: np.mean([records[time][key] for time in expected_times], axis=0)
                            for key in SPECTRAL_KEYS}
    audit = {
        "protocol_file_sha256": protocol_file_sha256, "sources": source_audit,
        "canonical_model_id": "M1", "repeatability_rtol": rtol, "repeatability_atol": atol,
        "truth_repeatability": "Exact equality required for identical physical targets.",
        "repeatability": repeated_audit,
    }
    return protocol, wavenumber, pooled, audit


def pooled_curves(pooled):
    """Construct skill from pooled sufficient statistics, never per-start ratios."""
    truth = pooled["M1"]["truth_power"]
    curves = {}
    for model_id, branch in (("GraphCast", "baseline"), *((model_id, "full") for model_id in MODEL_IDS)):
        record = pooled["M1" if model_id == "GraphCast" else model_id]
        power, error, cross = (record[f"{branch}_{suffix}"] for suffix in ("power", "error_power", "cross_power"))
        correlation = _divide(cross, np.sqrt(power) * np.sqrt(truth))
        finite = np.isfinite(correlation)
        _require(np.all(np.abs(correlation[finite]) <= 1 + 1e-6), f"{model_id}: pooled correlation outside [-1, 1]")
        curves[model_id] = {
            "power": power, "power_ratio": _divide(power, truth),
            "error_power": error, "error_ratio": _divide(error, truth),
            "cross_power": cross, "correlation": np.clip(correlation, -1, 1),
        }
    return truth, curves


def _write_csv(path, protocol, wavenumber, truth, curves):
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_id", "lead_hours", "variable", "zonal_wavenumber", "truth_power",
                         "prediction_power", "prediction_truth_power_ratio", "error_power",
                         "error_truth_power_ratio", "cospectrum", "spectral_correlation"])
        for model_id, metrics in curves.items():
            for step, lead in enumerate(protocol["lead_hours"]):
                for index, variable in enumerate(protocol["spectral_variables"]):
                    for mode, k in enumerate(wavenumber):
                        if k == 0:
                            continue
                        values = [truth[step, index, mode]] + [metrics[key][step, index, mode]
                            for key in ("power", "power_ratio", "error_power", "error_ratio", "cross_power", "correlation")]
                        writer.writerow([model_id, lead, variable, int(k),
                                         *[float(value) if np.isfinite(value) else "" for value in values]])


def _plot(image_dir, protocol, wavenumber, truth, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {model["model_id"]: model.get("label", model["model_id"]) for model in protocol["models"]}
    colors = {"GraphCast": "#666666", "M1": "#277da8", "M2": "#e1812c", "M3": "#8f5ca6"}
    specs = (
        ("power", "Zonal Fourier power", "spectra_power.png"),
        ("power_ratio", "Prediction / ERA5 power", "spectra_power_ratio.png"),
        ("error_ratio", "Forecast-error / ERA5 power", "spectra_error.png"),
        ("correlation", "Spectral correlation with ERA5", "spectra_correlation.png"),
    )
    image_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    variables = protocol["spectral_variables"]
    nonzero = wavenumber > 0
    for metric, title, filename in specs:
        height = 2.1 * len(variables) + 1.8
        fig, axes = plt.subplots(len(variables), len(DISPLAY_LEADS),
                                 figsize=(15, height), squeeze=False)
        for row, variable in enumerate(variables):
            for column, lead in enumerate(DISPLAY_LEADS):
                step = protocol["lead_hours"].index(lead)
                ax = axes[row, column]
                if metric == "power":
                    ax.plot(wavenumber[nonzero], truth[step, row, nonzero], color="#111111", lw=1.6, label="ERA5")
                for model_id, values in curves.items():
                    ax.plot(wavenumber[nonzero], values[metric][step, row, nonzero],
                            color=colors[model_id], lw=1.15, label=model_id)
                ax.set_xscale("log")
                if metric != "correlation":
                    ax.set_yscale("log")
                else:
                    ax.set_ylim(-1.02, 1.02)
                    ax.axhline(0, color="#aaaaaa", lw=.6)
                if metric in ("power_ratio", "correlation"):
                    ax.axhline(1, color="#aaaaaa", lw=.7, ls="--")
                ax.grid(True, alpha=.2, which="both")
                ax.tick_params(labelsize=8)
                if row == 0:
                    ax.set_title(f"Day {lead // 24}", fontsize=12)
                if column == 0:
                    unit = f"\n{VARIABLE_UNITS.get(variable, 'squared physical units')}" if metric == "power" else ""
                    ax.set_ylabel(variable + unit, fontsize=10)
                if row == len(variables) - 1:
                    ax.set_xlabel("Zonal wavenumber (cycles / circle)", fontsize=9)
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.suptitle(title, fontsize=18, y=1 - .12 / height)
        fig.text(.5, 1 - .51 / height,
                 f"Exploratory {_evaluation_year(protocol)} pilot · {len(protocol['initialization_times'])} starts · "
                 "30°–60° in both hemispheres · cosine-latitude weighting",
                 ha="center", fontsize=10)
        fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, 1 - .63 / height),
                   ncol=len(legend_labels), frameon=False)
        explanation = " | ".join(f"{key}: {labels[key]}" for key in MODEL_IDS)
        fig.text(.5, .28 / height, explanation, ha="center", fontsize=8)
        fig.text(.5, .1 / height, "Pooled power before ratios; zonal means/DC removed. No uncertainty intervals or preservation claim.",
                 ha="center", fontsize=9)
        fig.subplots_adjust(left=.08, right=.99, top=1 - 1.16 / height, bottom=.85 / height, hspace=.42, wspace=.32)
        path = image_dir / filename
        fig.savefig(path, dpi=150, facecolor="white")
        plt.close(fig)
        outputs.append(str(path.resolve()))
    return outputs


def report(workflow, *, make_plots=True):
    """Write complete-pilot CSV, provenance summary and four figures."""
    workflow = Path(workflow).resolve()
    protocol, wavenumber, pooled, audit = load_spectra(workflow)
    exact_loss, exact_loss_source = _validated_exact_loss(workflow, protocol)
    truth, curves = pooled_curves(pooled)
    data_dir = Path(protocol.get("output_data_dir", workflow / "report"))
    image_dir = Path(protocol.get("output_image_dir", workflow / "report"))
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "pooled_spectra.csv"
    _write_csv(csv_path, protocol, wavenumber, truth, curves)
    images = _plot(image_dir, protocol, wavenumber, truth, curves) if make_plots else []
    year = _evaluation_year(protocol)
    scope = protocol.get("scope", f"Exploratory {year} spatial Fourier pilot on fixed starts; no full-year or preservation claim")
    case_limitation = ("Exploratory 2022 development cases; not independent of model selection."
                       if year == 2022 else
                       f"Exploratory {year} pilot on fixed starts; not a full-year evaluation.")
    summary = {
        "status": "complete", "format": "v24_spectral_pilot_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": protocol["protocol_hash"], "initialization_times": protocol["initialization_times"],
        "evaluation_year": year, "evaluation_period": protocol.get("evaluation_period", f"{year} pilot"),
        "scope": scope,
        "samples_per_model": len(protocol["initialization_times"]), "target_steps": protocol["target_steps"],
        "lead_hours": protocol["lead_hours"], "spectral_variables": protocol["spectral_variables"],
        "model_ids": list(MODEL_IDS), "spectral_metadata": protocol["spectral_metadata"],
        "aggregation": "Equal-weight mean powers/cospectra across all starts before ratios; no DC in plots/CSV.",
        "denominators": "M1 canonical ERA5 power for all ratios and correlations; truth equality verified exactly.",
        "limitations": [case_limitation,
                        "Zonal Fourier spectra at 30–60 degrees latitude, not a global spherical-harmonic spectrum.",
                        "Wavenumber has no single latitude-independent physical wavelength.",
                        "No uncertainty intervals, extreme-event scores, or preservation/noninferiority conclusion."],
        "original_graphcast_loss": exact_loss, "exact_loss_validation_source": exact_loss_source,
        "audit": audit, "outputs": {"pooled_spectra_csv": str(csv_path.resolve()), "images": images},
    }
    _json_write(data_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    summary = report(args.workflow, make_plots=not args.no_plots)
    print(json.dumps({"status": summary["status"], "outputs": summary["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
