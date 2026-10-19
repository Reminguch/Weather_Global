#!/usr/bin/env python3
"""Validate and export saved four-model AMSE report data; no inference.

Run from the repository root after sourcing scripts/graphcast_env.sh.
All model arrays retain one explicitly selected run; repeated GC and MSE
evaluations are audit data and are never averaged into a displayed curve.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = "v24_amse_feature_report_20261004"
DEFAULT_OUTPUT = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
FULL_CAMPAIGN = "res1_amse_pair_20260919"
LAMBDA_CAMPAIGN = "res1_amse_blend_lambda0p1_20260924"
TAG = "swa_step00500-02000"
MODEL_ORDER = ("original_gc", "mse", "amse_lambda0p1", "full_amse")
MODELS = {
    "original_gc": {"label": "Original GC", "color": "#606871", "campaign": FULL_CAMPAIGN,
                    "arm": "mse_fp32", "branch": "baseline", "amse_lambda": None},
    "mse": {"label": "MSE", "color": "#0072B2", "campaign": FULL_CAMPAIGN,
            "arm": "mse_fp32", "branch": "full", "amse_lambda": None},
    "amse_lambda0p1": {"label": "AMSE λ=0.1", "color": "#009E73", "campaign": LAMBDA_CAMPAIGN,
                      "arm": "amse", "branch": "full", "amse_lambda": .1},
    "full_amse": {"label": "Full AMSE", "color": "#D55E00", "campaign": FULL_CAMPAIGN,
                  "arm": "amse", "branch": "full", "amse_lambda": 1.},
}
LEADS = np.arange(6, 241, 6)
DAYS = (1, 3, 7, 10)
VARIABLE_DEFINITIONS = {
    "2m_temperature": ("2 m temperature", "K", "K", 1.),
    "10m_u_component_of_wind": ("10 m zonal wind", "m/s", "m/s", 1.),
    "10m_v_component_of_wind": ("10 m meridional wind", "m/s", "m/s", 1.),
    "mean_sea_level_pressure": ("Mean sea-level pressure", "Pa", "hPa", .01),
    "total_precipitation_6hr": ("6 h precipitation", "m", "mm", 1000.),
    "temperature": ("Air temperature", "K", "K", 1.),
    "specific_humidity": ("Specific humidity", "kg/kg", "g/kg", 1000.),
    "u_component_of_wind": ("Zonal wind", "m/s", "m/s", 1.),
    "v_component_of_wind": ("Meridional wind", "m/s", "m/s", 1.),
    "geopotential": ("Geopotential", "m²/s²", "m²/s²", 1.),
    "vertical_velocity": ("Pressure vertical velocity", "Pa/s", "Pa/s", 1.),
}
FEATURE_SPECS = (
    ("T2m", "2 m temperature", "2m_temperature", None),
    ("U10", "10 m zonal wind", "10m_u_component_of_wind", None),
    ("V10", "10 m meridional wind", "10m_v_component_of_wind", None),
    ("MSLP", "Mean sea-level pressure", "mean_sea_level_pressure", None),
    ("TP6", "6 h precipitation", "total_precipitation_6hr", None),
    ("T850", "850 hPa temperature", "temperature", 850),
    ("Q700", "700 hPa specific humidity", "specific_humidity", 700),
    ("U850", "850 hPa zonal wind", "u_component_of_wind", 850),
    ("V850", "850 hPa meridional wind", "v_component_of_wind", 850),
    ("WS10", "10 m wind speed", None, None),
    ("WS850", "850 hPa wind speed", None, 850),
)
SPECTRAL_KEYS = ("truth_power", "baseline_power", "full_power", "baseline_cross_power",
                 "full_cross_power", "baseline_error_power", "full_error_power")


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Sources:
    def __init__(self):
        self.records = {}

    def add(self, path, expected=None, role="analysis_input"):
        path = Path(path).resolve()
        if path not in self.records:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            self.records[path] = {"path": str(path), "sha256": digest.hexdigest(),
                                  "size_bytes": path.stat().st_size, "roles": []}
        record = self.records[path]
        require(expected is None or record["sha256"] == expected, f"Source hash changed: {path}")
        if role not in record["roles"]:
            record["roles"].append(role)
        return {key: record[key] for key in ("path", "sha256", "size_bytes")}

    def read(self, path, expected=None, role="analysis_input"):
        self.add(path, expected, role)
        return json.loads(Path(path).read_text())


def feature_definitions():
    output = {}
    for code, label, variable, level in FEATURE_SPECS:
        _, units, display_units, scale = VARIABLE_DEFINITIONS[variable] if variable else (
            label, "m/s", "m/s", 1.)
        output[code] = {"label": label, "variable": variable, "pressure_level_hpa": level,
                        "metric_key": (f"{variable}_level{level}" if level else variable),
                        "units": units, "display_units": display_units, "display_scale": scale,
                        "spectral_available": True, "global_rmse_available": variable is not None,
                        "type": "derived_wind_speed" if variable is None else "primitive",
                        "definition": ("Wind speed is formed from physical wind components before the FFT. "
                                       "Saved global wind-speed RMSE is unavailable; component RMSEs cannot reconstruct it."
                                       if variable is None else label)}
    return output


def array40(value, label, positive=False):
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == (40,) and np.isfinite(result).all(), f"Invalid {label}")
    require(np.all(result > 0 if positive else result >= 0), f"Negative/zero {label}")
    return result


def validate_result(result, spec, count):
    for key in ("protocol_hash", "model_id", "checkpoint_sha256"):
        require(result[key] == spec[key], f"Result/specification {key} differs")
    require(result["baseline_checkpoint_sha256"] == spec["baseline"]["checkpoint_sha256"],
            "Baseline checkpoint differs")
    require(result["evaluation_status"] == "complete" and result["metrics_complete"] and
            result["evaluated_samples"] == count and result["remaining_samples"] == 0 and
            result["target_steps"] == 40, "Incomplete evaluation")
    require(result["resolution"] == 1 and result["eval_mode"] == "cold_full" and
            result["residual_state_init"] == "zero" and not result["rs_reset_every_step"],
            "Forecast protocol differs")
    require(len(result["per_channel_per_step"]) == 83, "Expected all 83 channels")
    require(set(result["per_variable_per_step"]) == set(VARIABLE_DEFINITIONS), "Variable set differs")
    for category in ("per_variable_per_step", "per_channel_per_step"):
        for name, metrics in result[category].items():
            for metric in ("rmse", "mae"):
                baseline = array40(metrics[f"{metric}_baseline"], f"{name} baseline {metric}", True)
                full = array40(metrics[f"{metric}_full"], f"{name} full {metric}")
                np.testing.assert_allclose(100 * (1 - full / baseline), metrics[f"improvement_pct_{metric}"],
                                           rtol=1e-10, atol=1e-10)
            for branch in ("baseline", "full"):
                require(np.all(np.asarray(metrics[f"mae_{branch}"]) <=
                               np.asarray(metrics[f"rmse_{branch}"]) * (1 + 2e-6)), "MAE exceeds RMSE")
    for variable, group in result["per_variable_per_step"].items():
        channels = [value for key, value in result["per_channel_per_step"].items()
                    if key.startswith(variable + "_level")]
        if channels:
            require(len(channels) == 13, f"Expected 13 levels: {variable}")
            for branch in ("baseline", "full"):
                np.testing.assert_allclose(np.sqrt(np.mean([np.asarray(c[f"rmse_{branch}"])**2
                                                            for c in channels], axis=0)),
                                           group[f"rmse_{branch}"], rtol=2e-6, atol=1e-12)
    loss = result["original_graphcast_loss"]
    baseline = array40(loss["baseline_per_step"], "GC baseline loss", True)
    full = array40(loss["full_per_step"], "GC full loss")
    for branch, values in (("baseline", baseline), ("full", full)):
        np.testing.assert_allclose(values.mean(), loss[f"{branch}_rollout"], rtol=1e-12)
    np.testing.assert_allclose(100*(1-full/baseline), loss["improvement_pct_per_step"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(100*(1-full.mean()/baseline.mean()), loss["improvement_pct_rollout"], atol=1e-10)
    if count == 32:
        require(result["chosen_idx"] == spec["chosen_idx"], "2022 anchor indices differ")
    else:
        require(result["global_initialization_times"] == spec["initialization_times"], "2023 dates differ")
        records = result["original_graphcast_loss_per_initialization"]
        require(len(records) == 8, "Expected eight per-start loss records")
        for branch in ("baseline", "full"):
            np.testing.assert_allclose(np.mean([row[f"{branch}_per_step"] for row in records], axis=0),
                                       loss[f"{branch}_per_step"], rtol=1e-12)


def branch_metrics(result, branch):
    loss = result["original_graphcast_loss"]
    return {"exact_graphcast_loss": {"per_step": loss[f"{branch}_per_step"],
                                     "rollout": loss[f"{branch}_rollout"]},
            **{category: {name: {metric: values[f"{metric}_{branch}"] for metric in ("rmse", "mae")}
                          for name, values in result[category].items()}
               for category in ("per_variable_per_step", "per_channel_per_step")}}


def difference(candidate, reference):
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    delta = candidate-reference
    return {"max_absolute_difference": float(np.max(np.abs(delta))),
            "mean_absolute_difference": float(np.mean(np.abs(delta))),
            "max_relative_difference_pct": float(100*np.max(np.abs(delta)/np.maximum(np.abs(reference), 1e-30)))}


def spectral_summary(P, T, C, E):
    P, T, C, E = (np.asarray(value)[1:] for value in (P, T, C, E))
    require(np.all(T > 0) and np.all(P > 0), "Undefined positive-wavenumber power ratio")
    coherence = C/np.sqrt(P*T)
    require(np.max(np.abs(coherence)) <= 1+1e-9, "Coherence out of bounds")
    amplitude = np.sum((np.sqrt(P)-np.sqrt(T))**2)
    decorrelation = np.sum(2*(np.sqrt(P*T)-C))
    total = np.sum(P+T-2*C)
    np.testing.assert_allclose(total, E.sum(), rtol=2e-10, atol=1e-10)
    ideal = np.sum(T*(1-np.maximum(np.clip(coherence, -1, 1), 0)**2))
    return {"mean_relative_power_error_pct": float(np.mean(100*np.abs(P-T)/T)),
            "mean_power_bias_pct": float(np.mean(100*(P/T-1))),
            "mean_coherence": float(coherence.mean()),
            "above_truth_power_bin_pct": float(100*np.mean(P>T)),
            "amplitude_mse": float(amplitude), "decorrelation_mse": float(decorrelation),
            "spectral_mse": float(total), "spectral_rmse": float(np.sqrt(total)),
            "ideal_rescaled_mse": float(ideal), "ideal_rescaled_rmse": float(np.sqrt(ideal)),
            "forecast_power_sum": float(P.sum()), "truth_power_sum": float(T.sum())}


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build(output=DEFAULT_OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = Sources()
    raw, specs, configs, pooled, truths_by_start = {}, {}, {}, {}, {}
    shared_spec, canonical_config = {}, None
    feature_order = [row[0] for row in FEATURE_SPECS]
    source_evaluations = []
    for campaign in (FULL_CAMPAIGN, LAMBDA_CAMPAIGN):
        experiment = ROOT/"artifacts/checkpoints/v24_Ilya"/campaign
        manifest = sources.read(experiment/"manifest.json", role="experiment_manifest")
        for filename in ("spectral_loss.py", "endpoint_step.py"):
            sources.add(experiment/"source/src/models/mamba/v24_Ilya/training"/filename,
                        role="frozen_training_loss_implementation")
        require(manifest["primary_swa_steps"] == [500, 1000, 1500, 2000], "SWA windows differ")
        for arm in ("mse_fp32", "amse"):
            key = (campaign, arm)
            config = sources.read(experiment/"runs"/arm/"run_config.json", role="run_configuration")
            configs[key] = config
            if canonical_config is None:
                canonical_config = config
            for field in ("architecture", "data", "optimizer", "sequence", "distributed", "memory", "validation"):
                require(config[field] == canonical_config[field], f"Matched config differs: {campaign}/{arm}/{field}")
            require(config["derived"]["parameter_overlay"]["init_from_sha256"] == manifest["init_sha256"],
                    "Initialization checkpoint identity differs")
            require(config["derived"]["parameter_overlay"]["init_from_sha256"] ==
                    canonical_config["derived"]["parameter_overlay"]["init_from_sha256"], "Campaign initializations differ")
            require(config["objective"]["loss_kind"] == arm, "Objective/arm mismatch")
            if arm == "amse":
                require(config["objective"].get("amse_lambda", 1.) == (1. if campaign == FULL_CAMPAIGN else .1),
                        "Unexpected AMSE interpolation")
            for stage, count in (("exact", 32), ("spectral", 8)):
                directory = experiment/"eval"/stage/arm/TAG
                spec = sources.read(directory/"specification.json", role="evaluation_specification")
                result = sources.read(directory/"evaluation.json", role="evaluation_metrics")
                # Protocol hash excludes its own field and uses sorted compact JSON.
                encoded = json.dumps({k: v for k, v in spec.items() if k != "protocol_hash"},
                                     sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                require(hashlib.sha256(encoded).hexdigest() == spec["protocol_hash"], "Specification hash mismatch")
                for item in (spec["run_config"], spec["experiment_manifest"], spec["data_metadata"], *spec["statistics"]):
                    sources.add(item["path"], item["sha256"], "frozen_provenance")
                sources.add(spec["checkpoint"], spec["checkpoint_sha256"], "evaluated_checkpoint")
                sources.add(spec["baseline"]["path"], spec["baseline"]["checkpoint_sha256"], "original_gc_checkpoint")
                for field in ("evaluation_reference", "initialization_manifest", "source_spectral_protocol"):
                    if field in spec:
                        sources.add(spec[field]["path"], spec[field]["sha256"], field)
                if stage not in shared_spec:
                    shared_spec[stage] = spec
                fields = ["baseline", "statistics", "architecture", "forecast", "data_path", "data_metadata",
                          "lead_hours", "target_steps", "evaluation_year", "n_samples", "warmup_steps"]
                fields += (["chosen_idx", "evaluation_reference"] if stage == "exact" else
                           ["initialization_times", "initialization_manifest", "spectral_variables", "spectral_metadata"])
                for field in fields:
                    require(spec[field] == shared_spec[stage][field], f"Evaluation scopes differ: {field}")
                validate_result(result, spec, count)
                raw[(campaign, arm, stage)] = result
                specs[(campaign, arm, stage)] = spec
                source_evaluations.append({"campaign": campaign, "arm": arm, "stage": stage,
                                           "path": str(directory/"evaluation.json"),
                                           "protocol_hash": spec["protocol_hash"],
                                           "checkpoint_sha256": spec["checkpoint_sha256"]})
                if stage != "spectral":
                    continue
                records = []
                paths = sorted(directory.glob("*.npz"))
                require(len(paths) == 8, "Expected eight complete spectral records")
                for path in paths:
                    sources.add(path, role="per_initialization_spectra")
                    with np.load(path, allow_pickle=False) as payload:
                        metadata = json.loads(payload["metadata"].item())
                        require(metadata["status"] == "complete", "Incomplete spectral record")
                        for field in ("protocol_hash", "checkpoint_sha256", "model_id"):
                            require(metadata[field] == spec[field], f"Spectral {field} mismatch")
                        require(metadata["baseline_checkpoint_sha256"] == spec["baseline"]["checkpoint_sha256"],
                                "Spectral GC checkpoint differs")
                        require(metadata["spectral_metadata"] == spec["spectral_metadata"], "Spectral scope differs")
                        require(metadata["initialization_time"] in spec["initialization_times"], "Unexpected spectral date")
                        np.testing.assert_array_equal(payload["lead_hours"], LEADS)
                        np.testing.assert_array_equal(payload["wavenumber"], np.arange(181))
                        require(list(payload["variables"]) == feature_order, "Spectral feature order differs")
                        arrays = {name: payload[name].copy() for name in SPECTRAL_KEYS}
                    for name, values in arrays.items():
                        require(values.shape == (40, 11, 181) and np.isfinite(values).all(), f"Invalid {name}")
                        if "cross" not in name:
                            require(np.all(values >= 0), f"Negative {name}")
                        require(np.all(values[..., 0] == 0), "Zonal mean must be removed")
                    for branch in ("baseline", "full"):
                        np.testing.assert_allclose(arrays[f"{branch}_power"]+arrays["truth_power"]-
                                                   2*arrays[f"{branch}_cross_power"], arrays[f"{branch}_error_power"],
                                                   rtol=1e-7, atol=1e-9)
                    stamp = metadata["initialization_time"]
                    if stamp in truths_by_start:
                        np.testing.assert_array_equal(arrays["truth_power"], truths_by_start[stamp])
                    else:
                        truths_by_start[stamp] = arrays["truth_power"]
                    records.append((stamp, arrays))
                require(sorted(stamp for stamp, _ in records) == sorted(spec["initialization_times"]),
                        "Missing/duplicate spectral dates")
                pooled[key] = {name: np.mean([arrays[name] for _, arrays in records], axis=0)
                               for name in SPECTRAL_KEYS}
        pooled_path = ROOT/"plots/analyze_models/data/resolution_eval"/campaign/"pooled_spectra.npz"
        sources.add(pooled_path, role="previous_pooled_spectral_crosscheck")
        with np.load(pooled_path, allow_pickle=False) as previous:
            for arm in ("mse_fp32", "amse"):
                for name, values in pooled[(campaign, arm)].items():
                    np.testing.assert_allclose(values, previous[f"{arm}_{TAG}_{name}"], rtol=1e-13, atol=1e-15)

    truth = pooled[(FULL_CAMPAIGN, "mse_fp32")]["truth_power"]
    arrays = {name: [] for name in ("power", "cross_power", "error_power")}
    for model in MODEL_ORDER:
        selected = MODELS[model]
        record = pooled[(selected["campaign"], selected["arm"])]
        np.testing.assert_array_equal(record["truth_power"], truth)
        for name in arrays:
            arrays[name].append(record[f"{selected['branch']}_{name}"])
    arrays = {name: np.stack(values) for name, values in arrays.items()}
    with (output/"spectra.npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, **arrays, truth_power=truth, lead_hours=LEADS,
                            variables=np.asarray(feature_order), wavenumber=np.arange(181),
                            model_ids=np.asarray(MODEL_ORDER))
    (output/"spectra.npz.tmp").replace(output/"spectra.npz")
    print(output/"spectra.npz", flush=True)

    models = json.loads(json.dumps(MODELS))
    evaluations, audit_rows, exact_rows, rmse_rows = {}, [], [], []
    for stage, name, count, year in (("exact", "exact32", 32, 2022), ("spectral", "spectral8", 8, 2023)):
        canonical = raw[(FULL_CAMPAIGN, "mse_fp32", stage)]
        canonical_gc = branch_metrics(canonical, "baseline")
        evaluation = {"year": year, "n_samples": count, "lead_hours": LEADS.tolist(), "models": {},
                      "origins": canonical["chosen_idx"] if stage == "exact" else canonical["global_initialization_times"]}
        for model in MODEL_ORDER:
            selection = models[model]
            key = (selection["campaign"], selection["arm"], stage)
            result, spec = raw[key], specs[key]
            row = branch_metrics(result, selection["branch"])
            row["paired_baseline"] = branch_metrics(result, "baseline")
            row["provenance"] = {"source_evaluation": next(v["path"] for v in source_evaluations
                                                           if (v["campaign"], v["arm"], v["stage"]) == key),
                                 "protocol_hash": spec["protocol_hash"], "selected_branch": selection["branch"],
                                 "checkpoint": spec["baseline"]["path"] if model == "original_gc" else spec["checkpoint"],
                                 "checkpoint_sha256": (spec["baseline"]["checkpoint_sha256"] if model == "original_gc"
                                                       else spec["checkpoint_sha256"])}
            selection.setdefault("provenance", {})[name] = row["provenance"]
            loss = row["exact_graphcast_loss"]
            reference = canonical_gc["exact_graphcast_loss"]
            paired = row["paired_baseline"]["exact_graphcast_loss"]
            loss["reduction_vs_canonical_gc_per_step_pct"] = (100*(1-np.asarray(loss["per_step"])/reference["per_step"])).tolist()
            loss["reduction_vs_canonical_gc_rollout_pct"] = 100*(1-loss["rollout"]/reference["rollout"])
            loss["reduction_vs_paired_gc_rollout_pct"] = 100*(1-loss["rollout"]/paired["rollout"])
            evaluation["models"][model] = row
            for li, lead in enumerate(LEADS):
                exact_rows.append({"evaluation": name, "model": model, "label": selection["label"],
                                   "lead_hours": int(lead), "loss": loss["per_step"][li],
                                   "canonical_gc_loss": reference["per_step"][li], "paired_gc_loss": paired["per_step"][li],
                                   "loss_reduction_vs_canonical_gc_pct": loss["reduction_vs_canonical_gc_per_step_pct"][li]})
            for category, kind in (("per_variable_per_step", "variable"), ("per_channel_per_step", "channel")):
                for channel, metrics in row[category].items():
                    variable = (channel.rsplit("_level", 1)[0]
                                if channel.rsplit("_level", 1)[-1].isdigit() else channel)
                    units = VARIABLE_DEFINITIONS[variable][1]
                    for li, lead in enumerate(LEADS):
                        rmse_rows.append({"evaluation": name, "model": model, "label": selection["label"],
                                          "kind": kind, "variable_or_channel": channel, "lead_hours": int(lead),
                                          "units": units, "rmse": metrics["rmse"][li], "mae": metrics["mae"][li],
                                          "paired_gc_rmse": row["paired_baseline"][category][channel]["rmse"][li]})
        evaluations[name] = evaluation
        for campaign in (FULL_CAMPAIGN, LAMBDA_CAMPAIGN):
            for arm in ("mse_fp32", "amse"):
                result = raw[(campaign, arm, stage)]
                repeated = branch_metrics(result, "baseline")
                audit_rows.append({"evaluation": name, "campaign": campaign, "arm": arm,
                                   "role": "repeated_original_gc", "plotted": campaign == FULL_CAMPAIGN and arm == "mse_fp32",
                                   "exact_graphcast_loss": repeated["exact_graphcast_loss"],
                                   "loss_drift_vs_canonical": difference(repeated["exact_graphcast_loss"]["per_step"],
                                                                         canonical_gc["exact_graphcast_loss"]["per_step"]),
                                   "per_channel_rmse_drift_vs_canonical": {
                                       channel: difference(metrics["rmse"], canonical_gc["per_channel_per_step"][channel]["rmse"])
                                       for channel, metrics in repeated["per_channel_per_step"].items()}})
        repeat = branch_metrics(raw[(LAMBDA_CAMPAIGN, "mse_fp32", stage)], "full")
        control = branch_metrics(canonical, "full")
        audit_rows.append({"evaluation": name, "campaign": LAMBDA_CAMPAIGN, "arm": "mse_fp32",
                           "role": "repeated_mse", "plotted": False, **repeat,
                           "loss_drift_vs_canonical_mse": difference(repeat["exact_graphcast_loss"]["per_step"],
                                                                    control["exact_graphcast_loss"]["per_step"]),
                           "per_channel_rmse_drift_vs_canonical_mse": {
                               channel: difference(metrics["rmse"], control["per_channel_per_step"][channel]["rmse"])
                               for channel, metrics in repeat["per_channel_per_step"].items()}})

    features = feature_definitions()
    spectral_rows = []
    for mi, model in enumerate(MODEL_ORDER):
        for vi, feature in enumerate(feature_order):
            for day in DAYS:
                li = day*4-1
                row = {"feature": feature, "model": model, "label": models[model]["label"],
                       "day": day, "lead_hours": day*24, "units": features[feature]["units"],
                       **spectral_summary(arrays["power"][mi, li, vi], truth[li, vi],
                                          arrays["cross_power"][mi, li, vi], arrays["error_power"][mi, li, vi])}
                spectral_rows.append(row)
    overall_spectral = {}
    for model in MODEL_ORDER:
        rows = [row for row in spectral_rows if row["model"] == model]
        overall_spectral[model] = {key: float(np.mean([r[key] for r in rows])) for key in
                                  ("mean_relative_power_error_pct", "mean_power_bias_pct", "mean_coherence", "above_truth_power_bin_pct")}
    # Audit the unused control spectra without introducing additional displayed models.
    spectral_audit = []
    for campaign in (FULL_CAMPAIGN, LAMBDA_CAMPAIGN):
        for arm in ("mse_fp32", "amse"):
            record = pooled[(campaign, arm)]
            spectral_audit.append({"campaign": campaign, "arm": arm, "role": "repeated_original_gc",
                                   "power_drift": difference(record["baseline_power"][..., 1:], arrays["power"][0, ..., 1:]),
                                   "cross_power_drift": difference(record["baseline_cross_power"][..., 1:], arrays["cross_power"][0, ..., 1:])})
    spectral_audit.append({"campaign": LAMBDA_CAMPAIGN, "arm": "mse_fp32", "role": "repeated_mse",
                           "power_drift": difference(pooled[(LAMBDA_CAMPAIGN, "mse_fp32")]["full_power"][..., 1:],
                                                     arrays["power"][1, ..., 1:])})
    definitions = {
        "global_rmse": "Square root of mean per-initialization global cosine-latitude-weighted MSE in physical units. Zonal means are included; this is not the mean of per-start RMSEs.",
        "variable_rmse": "Pressure-variable RMSE pools squared errors over all 13 levels equally before the square root. Per-channel metrics retain individual pressure levels.",
        "exact_graphcast_loss": "Original GraphCast weighted MSE: difference-standard-deviation normalization, exact latitude cell weights including poles, pressure-level and original variable weights. Rollout loss averages all 40 leads before computing reduction percentages.",
        "power": "One-sided longitude FFT power in squared physical units, pooled over eight starts, cosine-weighted latitudes 30–60 degrees in both hemispheres. Each latitude's zonal mean is removed; k=0 is zero.",
        "relative_power_error": "100*abs(forecast_power-truth_power)/truth_power. Equal-bin summaries weight all k=1..180, selected features and days equally; they are secondary diagnostics, not forecast-loss percentages.",
        "coherence": "Real pooled cross-power/sqrt(forecast_power*truth_power); measures pattern agreement including phase. Power and cross-power are pooled before forming ratios.",
        "spectral_mse": "Sum over k=1..180 of forecast_power+truth_power-2*cross_power; equals midlatitude zonal-anomaly MSE by Parseval. Its square root is spectral anomaly RMSE, distinct from global RMSE.",
        "amplitude_mse": "Sum_k(sqrt(forecast_power)-sqrt(truth_power))^2.",
        "decorrelation_mse": "Sum_k 2*(sqrt(forecast_power*truth_power)-cross_power). Both amplitude increases and coherence decreases can raise this term.",
        "ideal_rescaled_mse": "Sum_k truth_power*(1-max(coherence,0)^2): minimum MSE under independent nonnegative amplitude rescaling per k. Uses verification truth; in-sample diagnostic, not a validated forecast filter.",
        "amse_lambda": "Interpolation between the spherical spectral-MSE decomposition and full AMSE. λ=0 is spectral MSE, λ=1 is full AMSE; the training transform is global spherical T179 and differs from zonal verification.",
    }
    sources.add(Path(__file__), role="report_data_builder")
    for path in (ROOT/"src/models/mamba/v24_Ilya/metrics.py", ROOT/"src/models/mamba/v24_Ilya/spectral_diagnostics.py"):
        sources.add(path, role="metric_definition")
    document = {
        "schema_version": 1, "metadata": {"name": NAME, "analysis_date": "2026-10-04",
            "scope": "Saved 1° forecasts: 32 historical 2022 development starts and eight exploratory 2023 starts, 40 six-hour leads (10 days). No new model evaluation was run.",
            "model_policy": "Exactly four models. Original GC and MSE use the September 19 MSE evaluation; Full AMSE uses that campaign's AMSE evaluation; AMSE λ=0.1 uses the September 24 AMSE evaluation. Repeated controls are audited separately and never averaged into displayed curves.",
            "limitations": ["The 2022 starts and eight 2023 cases were previously inspected; these are development diagnostics, not an untouched test set.",
                            "No uncertainty intervals are available for this comparison.",
                            "Repeated original GC and MSE evaluations differ numerically; small cross-campaign differences require caution.",
                            "Global RMSE, exact GraphCast loss and midlatitude zonal spectra cover different quantities and regions.",
                            "Global RMSE is saved for 83 primitive channels and 11 variable groups. Global WS10/WS850 RMSE is unavailable; spectral wind-speed anomaly RMSE is available."],
            "checkpoint_policy": "Residual SWA averages updates 500, 1000, 1500 and 2000; cold full feedback, zero recurrent state.",
            "definitions": definitions, "spectral_protocol": shared_spec["spectral"]["spectral_metadata"],
            "training": {key: canonical_config[key] for key in ("architecture", "optimizer", "sequence")},
            "common_initialization_sha256": canonical_config["derived"]["parameter_overlay"]["init_from_sha256"]},
        "model_order": list(MODEL_ORDER), "models": models, "feature_order": feature_order, "features": features,
        "variable_definitions": {key: {"label": value[0], "units": value[1], "display_units": value[2], "display_scale": value[3]}
                                 for key, value in VARIABLE_DEFINITIONS.items()},
        "evaluations": evaluations,
        "summary": {"exact_graphcast_loss": {name: {model: row["exact_graphcast_loss"] for model, row in value["models"].items()}
                                                for name, value in evaluations.items()},
                    "spectral_overall_selected_days": overall_spectral, "spectral_by_feature_model_day": spectral_rows},
        "audit": {"baseline_policy": "Displayed Original GC is a single selected baseline. Saved paired baselines remain attached to each model for numerical-drift checks.",
                  "evaluations": audit_rows, "spectra": spectral_audit, "source_evaluations": source_evaluations,
                  "validated": ["Complete metrics and 40 leads for all eight source evaluations", "83 channel and 11 variable metrics",
                                "Same training architecture, initialization, data, optimizer, sequence and evaluation origins",
                                "Frozen specification/source/checkpoint hashes", "Same truth spectra at each initialization",
                                "Raw spectra reproduce saved pooled spectra", "Power/cross/error identities and saved metric improvement identities"]},
        "sources": list(sources.records.values()),
        "artifacts": {"spectra": str(output/"spectra.npz"), "spectral_summary_csv": str(output/"spectral_summary.csv"),
                      "global_metrics_csv": str(output/"global_rmse.csv"), "exact_loss_csv": str(output/"exact_graphcast_loss.csv")},
    }
    write_csv(output/"spectral_summary.csv", spectral_rows)
    write_csv(output/"global_rmse.csv", rmse_rows)
    write_csv(output/"exact_graphcast_loss.csv", exact_rows)
    (output/"report_data.json.tmp").write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    (output/"report_data.json.tmp").replace(output/"report_data.json")
    print(output/"report_data.json", flush=True)
    print(json.dumps({"models": len(MODEL_ORDER), "spectra_shape": arrays["power"].shape,
                      "source_files": len(sources.records), "rmse_rows": len(rmse_rows),
                      "spectral_summary_rows": len(spectral_rows), "exact_loss_rows": len(exact_rows)}), flush=True)
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    build(parser.parse_args().output_dir)
