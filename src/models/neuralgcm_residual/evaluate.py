"""Matched cold/warm forecasts, streaming paired scores and physical diagnostics."""
from __future__ import annotations

from pathlib import Path
import time
import jax
import numpy as np
from .config import FIELDS
from .data import STEP
from .finetune import burn_in
from .native_state import field_value, stop
from .model import zero_memory
from .loss import gaussian_weights, aggregate_reduction
from .io import append_jsonl, read_json, write_json, sha256, digest


def unit_metrics(prediction, target, latitude, levels, climatology):
    weights = gaussian_weights(latitude)[None, :]
    selected = {"T850": ("temperature", 850), "Z500": ("geopotential", 500),
                "Q700": ("specific_humidity", 700), "U850": ("u_component_of_wind", 850),
                "V850": ("v_component_of_wind", 850)}
    result = {}
    for label, (field, pressure) in selected.items():
        index = list(levels).index(pressure)
        p, t, c = (np.asarray(a[field][index], np.float64) for a in (prediction, target, climatology))
        area = lambda x: float((x * weights).sum(axis=-1).mean())
        mse = area((p - t) ** 2)
        numerator = area((p - c) * (t - c))
        denom = np.sqrt(area((p - c) ** 2) * area((t - c) ** 2))
        result[label] = {"mse": mse, "rmse": float(np.sqrt(mse)),
                         "acc": float(numerator / denom) if denom > 0 else None}
    for field in FIELDS[-2:]:
        error = np.asarray(prediction[field]) - np.asarray(target[field])
        result[field] = {"rmse": float(np.sqrt((error ** 2 * weights).sum(axis=-1).mean()))}
    return result


def native_diagnostics(runtime, state, reference):
    grid, model = runtime.adapter.grid, runtime.backbone.model
    weights = gaussian_weights(grid.latitudes)[None, None, :]
    pressure = lambda s: np.asarray(model.from_nondim_units(
        np.exp(np.asarray(grid.to_nodal(field_value(s, "log_surface_pressure")))), "Pa"))
    p, initial = pressure(state), pressure(reference)
    mean = lambda x: float((x * weights).sum(axis=-1).mean())
    result = {"mean_surface_pressure_pa": mean(p), "mass_relative_drift": mean(p) / mean(initial) - 1,
              "min_surface_pressure_pa": float(p.min()), "max_surface_pressure_pa": float(p.max())}
    for field in FIELDS[-3:]:
        q = np.asarray(grid.to_nodal(field_value(state, field)))
        result[field] = {"min": float(q.min()), "negative_fraction": float(np.mean(q < 0))}
    return result


def power_spectra(grid, prediction, target, baseline):
    result = {}
    # All seven fields, preserving level and total spherical wavenumber axes.
    for name in FIELDS:
        p, t, b = [np.asarray(grid.to_modal(value[name])) for value in (prediction, target, baseline)]
        power = lambda x: np.sum(np.square(x) * grid.mask, axis=-2)
        result[name] = {"prediction": power(p), "truth": power(t), "baseline": power(b),
                        "correction": power(p - b)}
    return result


def forecast_origin(runtime, params, origin, *, steps, warm=False, diagnostics=False, common=None):
    b, store, trainer = runtime.backbone, runtime.store, runtime.trainer
    memory = zero_memory(runtime.memory)
    key = jax.random.PRNGKey(22)
    if warm and params is not None:
        memory, key = burn_in(runtime.branch, params, memory, key, b, runtime.adapter,
                              runtime.normalization, runtime.known, store, origin)
    inputs, forcing = store.inputs_and_forcing(b.model, origin)
    initial = b.encode(inputs, forcing)
    physical, baseline = initial, initial
    corrected_scores, baseline_scores, details, spectra = [], [], [], []
    started = time.perf_counter()
    for lead in range(1, steps + 1):
        old = physical
        baseline = b.advance_6h(baseline, forcing)
        if params is None:
            physical = baseline
        else:
            prediction = b.advance_6h(old, forcing)
            key, subkey = jax.random.split(key)
            delta, memory = runtime.branch.apply(params, memory, subkey,
                runtime.normalization.inputs(runtime.adapter.features(old)), runtime.known(old, forcing))
            physical = runtime.adapter.apply_increment(prediction, runtime.normalization.increment(delta))
        b.assert_time(initial, physical, lead)
        decoded, base_decoded = b.decode(physical, forcing), b.decode(baseline, forcing)
        truth = store.frame(np.datetime64(origin, "h") + lead * STEP)
        score, base_score = float(trainer.weather_loss(decoded, truth)), float(trainer.weather_loss(base_decoded, truth))
        if not np.isfinite([score, base_score]).all():
            raise FloatingPointError(f"Nonfinite forecast at {origin}, lead {lead * 6}h")
        corrected_scores.append(score)
        baseline_scores.append(base_score)
        if diagnostics:
            detail = {"lead_hours": lead * 6, "physical": unit_metrics(decoded, truth,
                b.model.data_coords.horizontal.latitudes, b.model.data_coords.vertical.centers, runtime.climatology),
                "baseline_physical": unit_metrics(base_decoded, truth,
                b.model.data_coords.horizontal.latitudes, b.model.data_coords.vertical.centers, runtime.climatology),
                "native": native_diagnostics(runtime, physical, initial),
                "baseline_native": native_diagnostics(runtime, baseline, initial)}
            if common:
                regrid, loss = common
                map_fields = lambda x: {k: regrid(x[k]) for k in FIELDS}
                detail["common_grid_loss"] = float(loss(map_fields(decoded), map_fields(truth)))
                detail["common_grid_baseline_loss"] = float(loss(map_fields(base_decoded), map_fields(truth)))
            details.append(detail)
            spectra.append(power_spectra(b.model.data_coords.horizontal, decoded, truth, base_decoded))
    return {"origin": origin, "warm": warm, "steps": steps, "corrected": corrected_scores,
            "baseline": baseline_scores, "diagnostics": details, "seconds": time.perf_counter() - started}, spectra


def evaluate_origins(runtime, params, manifest, output, *, warm=False, diagnostics=False, common=None):
    import io
    from .io import atomic_bytes
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation output already exists")
    output.mkdir(parents=True, exist_ok=True)
    if manifest["dataset_id"] != runtime.store.identity:
        raise ValueError("Origin manifest does not match dataset")
    scores, failures, spectral_sums = [], [], {}
    for origin in manifest["origins"]:
        try:
            score, spectra = forecast_origin(runtime, params, origin, steps=manifest["steps"],
                warm=warm, diagnostics=diagnostics, common=common)
            scores.append(score)
            if diagnostics:
                for lead, fields in enumerate(spectra):
                    for field, quantities in fields.items():
                        for name, values in quantities.items():
                            key = f"lead{lead + 1:02d}__{field}__{name}"
                            spectral_sums[key] = spectral_sums.get(key, 0) + values
            append_jsonl(output / "origins.jsonl", score)
        except (FloatingPointError, ValueError) as exc:
            failure = {"origin": origin, "error": str(exc)}
            failures.append(failure)
            append_jsonl(output / "failures.jsonl", failure)
    summary = {"origin_manifest": digest(manifest), "warm": warm, "expected_origins": len(manifest["origins"]),
               "completed_origins": len(scores), "failures": failures,
               "loss_name": runtime.trainer.weather_loss.name,
               "eligible_for_selection": not failures and bool(scores)}
    if not failures and scores:
        c, b = np.array([s["corrected"] for s in scores]), np.array([s["baseline"] for s in scores])
        summary.update(loss=float(c.mean()), baseline_loss=float(b.mean()), reduction_pct=aggregate_reduction(c, b),
                       lead_loss=c.mean(axis=0).tolist(), baseline_lead_loss=b.mean(axis=0).tolist(),
                       five_day_loss=float(c[:, :20].mean()),
                       five_day_reduction_pct=aggregate_reduction(c[:, :20], b[:, :20]),
                       seconds=sum(s["seconds"] for s in scores))
        summary["block_bootstrap"] = block_bootstrap(scores)
    if spectral_sums:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **{k: v / len(scores) for k, v in spectral_sums.items()})
        atomic_bytes(output / "spectra.npz", buffer.getvalue())
    write_json(output / "summary.json", summary, immutable=True)
    return summary


def block_bootstrap(scores, repeats=1000, seed=22):
    # Paired calendar-week blocks keep dates together; no per-lead percentage average.
    blocks = {}
    for score in scores:
        week = int(np.datetime64(score["origin"], "D").astype(int)) // 7
        blocks.setdefault(week, []).append(score)
    keys, rng, estimates = list(blocks), np.random.default_rng(seed), []
    for _ in range(repeats):
        sample = [s for index in rng.integers(len(keys), size=len(keys)) for s in blocks[keys[index]]]
        estimates.append(aggregate_reduction([s["corrected"] for s in sample], [s["baseline"] for s in sample]))
    return {"method": "paired_calendar_week_blocks", "repeats": repeats,
            "reduction_pct_ci95": np.percentile(estimates, [2.5, 97.5]).tolist()}


def select_checkpoint(selection_path, checkpoint, summary, *, split, epoch_or_update):
    if split != "val" or summary["warm"] or not summary["eligible_for_selection"]:
        raise ValueError("Selection requires complete cold validation on 2022 origins")
    path = Path(selection_path)
    score = summary["five_day_loss"]
    previous = read_json(path) if path.exists() else None
    if previous is None or score < previous["score"]:
        metric = "cold_5day_" + summary.get("loss_name", "neuralgcm_field_normalized_mse")
        write_json(path, {"metric": metric, "split": "val",
                         "score": score, "checkpoint": str(Path(checkpoint).resolve()),
                         "sha256": sha256(checkpoint), "epoch_or_update": epoch_or_update,
                         "origin_manifest": summary["origin_manifest"]})
