#!/usr/bin/env python3
"""Evaluate the matched MSE/AMSE pair with independent, paired GC baselines.

Run in the GraphCast environment. ``evaluate --task 0|1`` builds the arm's SWA
and evaluates step 2000 and SWA on the historical 32 2022 anchors and the saved
eight 2023 spectrum starts. Every inference runs in a fresh subprocess.
``report`` writes exact loss, power, relative power error, and coherence plots.

The experiment manifest supplies ``workspace`` and two ``entries`` containing
``name``, ``config`` and optionally ``run_dir``. Optional overrides are
``evaluation_reference``, ``exact_data_path`` and ``spectral_protocol``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_v24_spectral_pilot import (  # noqa: E402
    check_file, frozen_file, protocol_digest, read_json, write_json,
)
from scripts.analyze_models.report_v24_spectral_pilot import (  # noqa: E402
    DISPLAY_LEADS, SPECTRAL_KEYS, VARIABLE_UNITS, _divide, _load_record, _require, _time,
)

ARMS = ("mse_fp32", "amse")
TAGS = ("step2000", "swa_step00500-02000")
SWA_STEPS = (500, 1000, 1500, 2000)
REFERENCE = ("artifacts/checkpoints/v22_final/res1_dm_7yr_k20_di_bcg_20k_20260813/"
             "di16_bcg1_closed_sg_stateful_20k/eval/cold_full_zero_exact/swa_step02000-08000.json")
PILOT = "plots/analyze_models/data/resolution_eval/res1_spectral_pilot_2023_20260917/protocol.json"
EXACT_DATA = ("/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/"
              "wb2_res1_levels13_2015_2022.zarr")


def resolve_path(path, workspace):
    value = Path(path)
    return value if value.is_absolute() else workspace / value


def load_experiment(experiment):
    manifest = read_json(experiment / "manifest.json")
    workspace = Path(manifest.get("workspace", ROOT)).resolve()
    entries = manifest["entries"]
    _require([entry["name"] for entry in entries] == list(ARMS),
             "Expected ordered mse_fp32 and amse manifest entries")
    return manifest, workspace


def checkpoint_paths(experiment, entry):
    directory = Path(entry.get("run_dir", experiment / "runs" / entry["name"]))
    return {
        "step2000": directory / "checkpoints/checkpoint_step00002000.pkl",
        "swa_step00500-02000": directory / "swa/swa_step00500-02000.pkl",
    }


def ensure_swa(experiment, entry):
    from src.models.mamba.v24_Ilya.checkpoint import build_v24_Ilya_swa

    paths = checkpoint_paths(experiment, entry)
    output = paths[TAGS[1]]
    if not output.exists():
        sources = [paths[TAGS[0]].parent / f"checkpoint_step{step:08d}.pkl" for step in SWA_STEPS]
        build_v24_Ilya_swa(sources, list(SWA_STEPS), output)


def make_specification(experiment, task, tag, stage):
    manifest, workspace = load_experiment(experiment)
    _require(0 <= task < len(ARMS), "Invalid arm task")
    _require(tag in TAGS and stage in ("exact", "spectral"), "Invalid evaluation stage/tag")
    entry = manifest["entries"][task]
    run_dir = Path(entry.get("run_dir", experiment / "runs" / entry["name"]))
    # The resolved configuration is the actual training configuration.
    configuration = frozen_file(run_dir / "run_config.json")
    cfg = read_json(configuration["path"])
    baseline = frozen_file(resolve_path(cfg["data"]["baseline_checkpoint"], workspace))
    stats_dir = resolve_path(cfg["data"]["stats_dir"], workspace)
    ckpt = frozen_file(checkpoint_paths(experiment, entry)[tag])
    spec = {
        "schema_version": 1, "stage": stage, "arm": entry["name"], "tag": tag,
        "model_id": f"{entry['name']}_{tag}", "checkpoint": ckpt["path"],
        "checkpoint_sha256": ckpt["sha256"], "run_config": configuration,
        "experiment_manifest": frozen_file(experiment / "manifest.json"),
        "architecture": cfg["architecture"], "target_steps": 40,
        "lead_hours": list(range(6, 241, 6)),
        "baseline": {"path": baseline["path"], "checkpoint_sha256": baseline["sha256"]},
        "stats_dir": str(stats_dir),
        "statistics": [frozen_file(stats_dir / name) for name in
                       ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc")],
        "forecast": {"eval_mode": "cold_full", "residual_state_init": "zero", "residual_alpha": 1.,
                     "seed": 0, "input_duration": "12h"},
    }
    if stage == "exact":
        reference = frozen_file(resolve_path(manifest.get("evaluation_reference", REFERENCE), workspace))
        historical = read_json(reference["path"])
        _require(len(historical["chosen_idx"]) == 32, "Reference must have 32 anchors")
        spec.update(data_path=str(resolve_path(manifest.get("exact_data_path", EXACT_DATA), workspace)),
                    evaluation_year=2022, n_samples=32, warmup_steps=24,
                    evaluation_reference=reference, chosen_idx=historical["chosen_idx"])
    else:
        protocol_file = frozen_file(resolve_path(manifest.get("spectral_protocol", PILOT), workspace))
        pilot = read_json(protocol_file["path"])
        _require(protocol_digest(pilot) == pilot["protocol_hash"], "Saved spectral protocol hash mismatch")
        _require(pilot["evaluation_year"] == 2023 and len(pilot["initialization_times"]) == 8,
                 "Expected the saved eight 2023 spectral starts")
        _require(pilot["baseline"]["checkpoint_sha256"] == baseline["sha256"],
                 "Training baseline differs from original spectral pilot GC")
        check_file(pilot["initialization_manifest"])
        check_file(pilot["data_metadata"])
        spec.update(data_path=pilot["data_path"], evaluation_year=2023, n_samples=8, warmup_steps=0,
                    source_spectral_protocol=protocol_file,
                    initialization_times=pilot["initialization_times"],
                    initialization_manifest=pilot["initialization_manifest"],
                    initialization_manifest_sha256=pilot["initialization_manifest_sha256"],
                    spectral_variables=pilot["spectral_variables"], spectral_metadata=pilot["spectral_metadata"])
    spec["data_metadata"] = frozen_file(Path(spec["data_path"]) / ".zmetadata")
    spec["protocol_hash"] = protocol_digest(spec)
    return spec


def verify_specification(spec):
    _require(protocol_digest(spec) == spec["protocol_hash"], "Evaluation specification hash mismatch")
    records = [spec["run_config"], spec["experiment_manifest"], spec["data_metadata"], *spec["statistics"],
               {"path": spec["checkpoint"], "sha256": spec["checkpoint_sha256"]},
               {"path": spec["baseline"]["path"], "sha256": spec["baseline"]["checkpoint_sha256"]}]
    for key in ("evaluation_reference", "initialization_manifest", "source_spectral_protocol"):
        if key in spec:
            records.append(spec[key])
    for record in records:
        check_file(record)


def exact_arrays(result, spec):
    """Validate completeness and provenance without requiring baseline agreement."""
    _require(result.get("evaluation_status") == "complete", "Evaluation is incomplete")
    for key in ("protocol_hash", "model_id", "checkpoint_sha256"):
        _require(result.get(key) == spec[key], f"Evaluation {key} mismatch")
    _require(result.get("baseline_checkpoint_sha256") == spec["baseline"]["checkpoint_sha256"],
             "Baseline checkpoint mismatch")
    _require(result.get("target_steps") == 40 and result.get("evaluated_samples") == spec["n_samples"],
             "Evaluation sample/lead coverage mismatch")
    if spec["stage"] == "exact":
        _require(result["chosen_idx"] == spec["chosen_idx"], "2022 anchors differ from matched reference")
    else:
        records = result["original_graphcast_loss_per_initialization"]
        _require([_time(row["initialization_time"]) for row in records] ==
                 [_time(stamp) for stamp in spec["initialization_times"]], "Spectral dates differ")
        _require(result["initialization_manifest_sha256"] == spec["initialization_manifest_sha256"],
                 "Spectral initialization manifest mismatch")
    loss = result["original_graphcast_loss"]
    arrays = {}
    for branch in ("baseline", "full"):
        values = np.asarray(loss[f"{branch}_per_step"], dtype=np.float64)
        _require(values.shape == (40,) and np.isfinite(values).all() and np.all(values >= 0),
                 f"Invalid {branch} exact loss")
        if branch == "baseline":
            _require(np.all(values > 0), "Zero baseline exact loss")
        np.testing.assert_allclose(values.mean(), loss[f"{branch}_rollout"], rtol=1e-6, atol=1e-7)
        if spec["stage"] == "spectral":
            per_start = np.asarray([row[f"{branch}_per_step"] for row in records], dtype=np.float64)
            _require(per_start.shape == (8, 40) and np.isfinite(per_start).all() and np.all(per_start >= 0),
                     f"Invalid per-start {branch} losses")
            np.testing.assert_allclose(per_start.mean(axis=0), values, rtol=1e-6, atol=1e-7)
        arrays[branch] = values
    return arrays


def evaluate_worker(experiment, task, tag, stage):
    import jax
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    from src.models.mamba.v24_Ilya.spectral_observer import SpectralObserver

    _require(any(device.platform == "gpu" for device in jax.devices()), "Evaluation needs a GPU allocation")
    spec = make_specification(experiment, task, tag, stage)
    destination = experiment / "eval" / stage / spec["arm"] / tag
    output = destination / "evaluation.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    destination.mkdir(parents=True, exist_ok=True)
    specification = destination / "specification.json"
    if specification.exists():
        _require(read_json(specification) == spec, "Existing evaluation specification differs")
    else:
        write_json(specification, spec)
    verify_specification(spec)
    identity = {key: spec[key] for key in ("protocol_hash", "model_id", "checkpoint_sha256")}
    identity["baseline_checkpoint_sha256"] = spec["baseline"]["checkpoint_sha256"]
    observer = (SpectralObserver(destination, identity, target_steps=40, export_fields=False)
                if stage == "spectral" else None)
    unpublished = destination / "evaluation.unpublished.json"
    cfg = V24IlyaEvalConfig(
        ckpt=Path(spec["checkpoint"]), out_json=unpublished,
        ckpt_in=Path(spec["baseline"]["path"]), stats_dir=Path(spec["stats_dir"]),
        data_path=spec["data_path"], val_year=spec["evaluation_year"],
        train_start_year=2015, train_end_year=2021, target_steps=40,
        n_samples=spec["n_samples"], warmup_steps=spec["warmup_steps"], omit_rms_bias=True,
        initialization_manifest=(Path(spec["initialization_manifest"]["path"]) if stage == "spectral" else None),
        model_id=spec["model_id"], **spec["forecast"], **spec["architecture"],
    )
    started, status = time.monotonic(), "failed"
    try:
        result = evaluate_v24_Ilya(cfg, prediction_observer=observer)
        result.update(identity)
        exact_arrays(result, spec)
        write_json(output, result)
        unpublished.unlink(missing_ok=True)
        status = "complete"
    finally:
        write_json(destination / "resources.json", {
            **identity, "status": status, "elapsed_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            "devices": [{"platform": d.platform, "kind": d.device_kind} for d in jax.devices()],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        })


def evaluate(experiment, task, stage="evaluate"):
    manifest, _ = load_experiment(experiment)
    _require(0 <= task < len(ARMS), "Invalid arm task")
    # Keep checkpoint loading outside GPU workers; it can hold several large
    # optimizer trees. Release that process before allocating model memory.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "swa", "--experiment", str(experiment),
                    "--task", str(task)], check=True)
    for selected in (("exact", "spectral") if stage == "evaluate" else (stage,)):
        for tag in TAGS:
            output = experiment / "eval" / selected / manifest["entries"][task]["name"] / tag / "evaluation.json"
            if output.exists():
                spec = make_specification(experiment, task, tag, selected)
                exact_arrays(read_json(output), spec)
                print(f"Reusing complete evaluation: {output}", flush=True)
                continue
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--experiment", str(experiment),
                            "--task", str(task), "--tag", tag, "--stage", selected], check=True)


def smoke(experiment, task, checkpoint):
    """Round-trip a native smoke checkpoint through two physical rollout steps."""
    import jax
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya

    _require(any(device.platform == "gpu" for device in jax.devices()), "Smoke evaluation needs a GPU")
    manifest, workspace = load_experiment(experiment)
    entry = manifest["entries"][task]
    cfg = read_json(resolve_path(entry["config"], workspace))
    pilot = read_json(resolve_path(manifest.get("spectral_protocol", PILOT), workspace))
    _require(protocol_digest(pilot) == pilot["protocol_hash"], "Saved spectral protocol hash mismatch")
    directory = experiment / "checks"
    output = directory / f"{entry['name']}_roundtrip.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    initializations = directory / f"{entry['name']}_roundtrip_initializations.json"
    write_json(initializations, {"schema_version": 1, "initializations": [
        {"initialization_time": pilot["initialization_times"][0]}]})
    baseline = resolve_path(cfg["data"]["baseline_checkpoint"], workspace)
    baseline_identity = frozen_file(baseline)
    _require(baseline_identity["sha256"] == pilot["baseline"]["checkpoint_sha256"],
             "Smoke baseline differs from original spectral pilot GC")
    checkpoint_identity = frozen_file(checkpoint)
    unpublished = directory / f"{entry['name']}_roundtrip.unpublished.json"
    config = V24IlyaEvalConfig(
        ckpt=checkpoint, out_json=unpublished, ckpt_in=baseline,
        stats_dir=resolve_path(cfg["data"]["stats_dir"], workspace), data_path=pilot["data_path"],
        val_year=2023, train_start_year=2015, train_end_year=2021,
        eval_mode="cold_full", residual_state_init="zero", residual_alpha=1.,
        target_steps=2, warmup_steps=0, n_samples=1, seed=0, omit_rms_bias=True,
        initialization_manifest=initializations, model_id=f"{entry['name']}_roundtrip",
        **cfg["architecture"],
    )
    started = time.monotonic()
    result = evaluate_v24_Ilya(config)
    _require(result.get("evaluation_status") == "complete" and result.get("evaluated_samples") == 1,
             "Smoke round-trip evaluation incomplete")
    for branch in ("baseline", "full"):
        values = np.asarray(result["original_graphcast_loss"][f"{branch}_per_step"])
        _require(values.shape == (2,) and np.isfinite(values).all() and np.all(values >= 0),
                 f"Invalid smoke round-trip {branch} exact losses")
    result.update(passed=True, checkpoint=checkpoint_identity, baseline_checkpoint=baseline_identity,
                  elapsed_seconds=time.monotonic() - started,
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    write_json(output, result)
    unpublished.unlink(missing_ok=True)
    print(output, flush=True)


def baseline_drift(candidate, reference, *, rtol=1e-3, atol=1e-6):
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    _require(candidate.shape == reference.shape, "Baseline drift arrays have different shapes")
    _require(np.isfinite(candidate).all() and np.isfinite(reference).all(), "Nonfinite baseline drift input")
    difference = np.abs(candidate - reference)
    relative = _divide(difference, np.abs(reference))
    positive = np.abs(reference) > 0
    return {"rtol": rtol, "atol": atol, "values": int(candidate.size),
            "outside_tolerance": int(np.count_nonzero(difference > atol + rtol * np.abs(reference))),
            "max_absolute_difference": float(difference.max()),
            "max_relative_difference_pct": float(100 * relative[positive].max()) if positive.any() else None,
            "nonzero_candidate_at_zero_reference": int(np.count_nonzero(~positive & (candidate != 0)))}


def spectral_curves(record):
    truth = record["truth_power"]
    result = {}
    for branch in ("baseline", "full"):
        power = record[f"{branch}_power"]
        coherence = _divide(record[f"{branch}_cross_power"], np.sqrt(power) * np.sqrt(truth))
        finite = np.isfinite(coherence)
        _require(np.all(np.abs(coherence[finite]) <= 1 + 1e-6), "Coherence outside [-1, 1]")
        result[branch] = {
            "power": power,
            "relative_power_error_pct": 100 * _divide(np.abs(power - truth), truth),
            "power_bias_pct": 100 * (_divide(power, truth) - 1),
            "coherence": np.clip(coherence, -1, 1),
        }
    return result


def load_results(experiment):
    manifest, _ = load_experiment(experiment)
    exact, pooled, sources, specifications, audit = {}, {}, [], {}, {}
    truth_records = None
    baseline_records = None
    wavenumber = None
    for stage in ("exact", "spectral"):
        canonical_baseline = None
        for entry in manifest["entries"]:
            for tag in TAGS:
                model_id = f"{entry['name']}_{tag}"
                directory = experiment / "eval" / stage / entry["name"] / tag
                spec = read_json(directory / "specification.json")
                verify_specification(spec)
                _require(spec["arm"] == entry["name"] and spec["tag"] == tag and spec["stage"] == stage,
                         "Evaluation path and specification disagree")
                result = read_json(directory / "evaluation.json")
                values = exact_arrays(result, spec)
                key = f"{stage}/{model_id}"
                sources.extend(frozen_file(directory / name) for name in ("evaluation.json", "specification.json"))
                specifications[key] = spec
                if canonical_baseline is None:
                    canonical_baseline = values["baseline"]
                drift = baseline_drift(values["baseline"], canonical_baseline)
                exact[key] = {"stage": stage, "arm": entry["name"], "tag": tag,
                              "baseline_per_step": values["baseline"].tolist(), "full_per_step": values["full"].tolist(),
                              "baseline_rollout": float(values["baseline"].mean()),
                              "full_rollout": float(values["full"].mean()),
                              "improvement_pct_rollout": float(100 * (1 - values["full"].mean() / values["baseline"].mean())),
                              "improvement_pct_per_step": (100 * (1 - values["full"] / values["baseline"])).tolist(),
                              "baseline_drift_vs_mse_step2000": drift}
                if stage == "exact":
                    continue
                times = [_time(stamp) for stamp in spec["initialization_times"]]
                records = {}
                for path in sorted(directory.glob("*.npz")):
                    stamp, wavenumber, arrays, provenance = _load_record(path, spec, spec, times, wavenumber)
                    _require(stamp not in records, "Duplicate spectral initialization")
                    records[stamp] = arrays
                    sources.append(provenance)
                _require(set(records) == set(times), f"Incomplete spectral initializations: {model_id}")
                if truth_records is None:
                    truth_records = records
                    baseline_records = records
                for stamp in times:
                    _require(np.array_equal(records[stamp]["truth_power"], truth_records[stamp]["truth_power"]),
                             "ERA5 truth differs between evaluations")
                pooled[model_id] = {name: np.mean([records[t][name] for t in times], axis=0) for name in SPECTRAL_KEYS}
                audit[model_id] = {name: baseline_drift(
                    np.stack([records[t][name][..., 1:] for t in times]),
                    np.stack([baseline_records[t][name][..., 1:] for t in times]), atol=1e-10)
                    for name in ("baseline_power", "baseline_cross_power", "baseline_error_power")}
    _require(len({spec["baseline"]["checkpoint_sha256"] for spec in specifications.values()}) == 1,
             "Evaluations use different GC checkpoints")
    return exact, pooled, wavenumber, specifications, audit, sources


def plot_report(image_dir, exact, pooled, wavenumber, spec, drift_detected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    colors = {"mse_fp32": "#0072B2", "amse": "#D55E00"}
    note = ("Repeated GC baselines differ beyond tolerance; each paired baseline is shown."
            if drift_detected else "Each trained forecast is compared with its own repeated original GC baseline.")
    curves = {name: spectral_curves(record) for name, record in pooled.items()}
    variables = spec["spectral_variables"]
    for metric, ylabel in (("power", "Power"), ("relative_power_error_pct", "Relative power-spectrum error (%)"),
                           ("coherence", "Forecast–truth spectral coherence")):
        with PdfPages(image_dir / f"{metric}.pdf") as pdf:
            for tag in TAGS:
                for vi, variable in enumerate(variables):
                    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
                    for ax, lead in zip(axes.ravel(), DISPLAY_LEADS, strict=True):
                        li = spec["lead_hours"].index(lead)
                        if metric == "power":
                            ax.plot(wavenumber[1:], pooled[f"mse_fp32_{tag}"]["truth_power"][li, vi, 1:],
                                    color="black", lw=1.8, label="ERA5")
                            ax.set_yscale("log")
                        elif metric == "coherence":
                            ax.axhline(1, color="black", lw=.8)
                            ax.set_ylim(-1.03, 1.03)
                        else:
                            ax.axhline(0, color="black", lw=.8)
                        for ai, arm in enumerate(ARMS):
                            for branch in ("baseline", "full"):
                                ax.plot(wavenumber[1:], curves[f"{arm}_{tag}"][branch][metric][li, vi, 1:],
                                        color=colors[arm] if branch == "full" else ("#666666", "#999999")[ai],
                                        ls="-" if branch == "full" else ("--", ":")[ai], lw=1.4,
                                        label=arm if branch == "full" else f"Original GC, {arm} evaluation")
                        ax.set_xscale("log")
                        ax.tick_params(labelbottom=True)
                        ax.set_title(f"Day {lead // 24}")
                        ax.set_xlabel("Zonal wavenumber (cycles / latitude circle)")
                        ax.set_ylabel(f"Power ({VARIABLE_UNITS[variable]})" if metric == "power" else ylabel)
                        ax.grid(alpha=.2, which="both")
                    if metric == "relative_power_error_pct":
                        axes[0, 0].set_ylim(bottom=0)
                    fig.suptitle(f"{variable}: {ylabel.lower()} · {tag}", y=.975, fontsize=15)
                    formula = {"power": "Squared Fourier amplitudes, averaged over latitude and starts",
                               "relative_power_error_pct": "100 × |forecast power − ERA5 power| / ERA5 power · Lower is better",
                               "coherence": "Real cospectrum / sqrt(forecast power × ERA5 power) · Higher is better"}[metric]
                    fig.text(.5, .932, formula, ha="center", fontsize=10)
                    handles, labels = axes[0, 0].get_legend_handles_labels()
                    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .074), ncol=2, frameon=False)
                    fig.text(.5, .048, "1° · Eight fixed 2023 starts · 30–60° both hemispheres · Pooled powers before ratios",
                             ha="center", fontsize=9)
                    fig.text(.5, .025, note, ha="center", fontsize=9)
                    fig.subplots_adjust(top=.855, bottom=.25, left=.095, right=.97, hspace=.43, wspace=.24)
                    pdf.savefig(fig)
                    if variable == "T850":
                        fig.savefig(image_dir / f"{metric}_{tag}_T850.png", dpi=160)
                    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ai, stage in enumerate(("exact", "spectral")):
        for ti, tag in enumerate(TAGS):
            ax = axes[ai, ti]
            for arm in ARMS:
                row = exact[f"{stage}/{arm}_{tag}"]
                ax.plot(np.arange(1, 41) / 4, row["improvement_pct_per_step"], label=arm, color=colors[arm])
            ax.axhline(0, color="black", lw=.8)
            ax.grid(alpha=.2)
            ax.set_title(f"{'32 starts, 2022' if stage == 'exact' else '8 starts, 2023'} · {tag}")
            ax.set_xlabel("Forecast lead (days)")
            ax.set_ylabel("Exact GraphCast loss reduction (%)")
    fig.suptitle("Exact GraphCast loss reduction versus each paired original GC baseline", y=.975)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .06), ncol=2, frameon=False)
    fig.text(.5, .025, note, ha="center", fontsize=9)
    fig.subplots_adjust(top=.9, bottom=.19, left=.095, right=.97, hspace=.35, wspace=.25)
    fig.savefig(image_dir / "exact_graphcast_loss.pdf")
    fig.savefig(image_dir / "exact_graphcast_loss.png", dpi=160)
    plt.close(fig)


def report(experiment, *, make_plots=True):
    _, workspace = load_experiment(experiment)
    exact, pooled, wavenumber, specs, audit, sources = load_results(experiment)
    data_dir = workspace / "plots/analyze_models/data/resolution_eval" / experiment.name
    image_dir = workspace / "plots/analyze_models/images/resolution_eval" / experiment.name
    data_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    drift_detected = any(row["baseline_drift_vs_mse_step2000"]["outside_tolerance"] for row in exact.values())
    drift_detected |= any(row["outside_tolerance"] for values in audit.values() for row in values.values())
    summary = {"status": "complete_with_baseline_drift" if drift_detected else "complete",
               "baseline_policy": "Each model uses its own paired original GC; repeated baselines are never silently merged.",
               "scope": "32 matched 2022 starts plus exploratory eight-start 2023 zonal spectra; no uncertainty intervals.",
               "exact_graphcast_loss": exact, "spectral_baseline_drift_vs_mse_step2000": audit,
               "evaluation_sources": sources, "checkpoint_mapping": {key: {name: value[name] for name in
                   ("checkpoint", "checkpoint_sha256", "run_config")} for key, value in specs.items()},
               "data_dir": str(data_dir), "image_dir": str(image_dir)}
    write_json(data_dir / "comparison.json", summary)
    with (data_dir / "exact_graphcast_loss.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "arm", "checkpoint", "lead_hours", "baseline_loss", "model_loss", "loss_reduction_pct"])
        for row in exact.values():
            for li in range(40):
                writer.writerow([row["stage"], row["arm"], row["tag"], 6 * (li + 1),
                                 row["baseline_per_step"][li], row["full_per_step"][li], row["improvement_pct_per_step"][li]])
    spec = specs[f"spectral/mse_fp32_{TAGS[0]}"]
    np.savez_compressed(data_dir / "pooled_spectra.npz", wavenumber=wavenumber,
                        lead_hours=spec["lead_hours"], variables=spec["spectral_variables"],
                        **{f"{model}_{key}": values for model, arrays in pooled.items() for key, values in arrays.items()})
    with (data_dir / "spectral_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "branch", "variable", "lead_hours", "wavenumber", "power", "relative_power_error_pct", "coherence"])
        for model, record in pooled.items():
            for branch, curves in spectral_curves(record).items():
                for lead in DISPLAY_LEADS:
                    li = spec["lead_hours"].index(lead)
                    for vi, variable in enumerate(spec["spectral_variables"]):
                        for ki in range(1, len(wavenumber)):
                            writer.writerow([model, branch, variable, lead, wavenumber[ki],
                                             *[curves[key][li, vi, ki] for key in ("power", "relative_power_error_pct", "coherence")]])
    if make_plots:
        plot_report(image_dir, exact, pooled, wavenumber, spec, drift_detected)
    lines = ["# Matched MSE versus full-forecast AMSE", "", summary["scope"], "", summary["baseline_policy"], "",
             f"Baseline drift detected: **{bool(drift_detected)}**. See comparison.json for all tolerances and differences.", "",
             "| Evaluation | Arm | Checkpoint | Exact GC rollout loss reduction |", "|---|---|---|---:|"]
    lines.extend(f"| {row['stage']} | {row['arm']} | {row['tag']} | {row['improvement_pct_rollout']:.3f}% |" for row in exact.values())
    lines += ["", "Powers and cospectra are averaged over starts before ratios. Relative power error is "
              "100 × |forecast power − truth power| / truth power; coherence is the real cospectrum divided "
              "by the geometric mean of forecast and truth powers. Zero-power ratios are undefined (omitted).",
              "", "Training uses global spherical AMSE; these zonal diagnostics retain the previous pilot's "
              "30–60° latitude bands and are a separate verification measure.", "",
              "SWA averages residual parameters from updates 500, 1000, 1500 and 2000. All evaluations use "
              "cold full feedback, zero recurrent state, and the frozen original GraphCast checkpoint.", "",
              f"Data: {data_dir}", "", f"Plots: {image_dir}", "", "## Checkpoints", ""]
    lines.extend(f"- `{key}`: `{value['checkpoint']}` (SHA256 `{value['checkpoint_sha256']}`)" for key, value in specs.items())
    text = "\n".join(lines) + "\n"
    (data_dir / "README.md").write_text(text)
    destination = experiment / "eval/report"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "README.md").write_text(text)
    write_json(destination / "comparison.json", summary)
    print(data_dir / "README.md", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("evaluate", "exact", "spectral", "report", "worker", "swa", "smoke"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--task", type=int)
    parser.add_argument("--tag", choices=TAGS)
    parser.add_argument("--stage", choices=("exact", "spectral"))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    experiment = args.experiment.resolve()
    if args.phase == "report":
        report(experiment)
        return
    if args.task not in (0, 1):
        parser.error("--task 0 or 1 is required outside report")
    if args.phase == "smoke":
        if args.checkpoint is None:
            parser.error("smoke requires --checkpoint")
        smoke(experiment, args.task, args.checkpoint.resolve())
    elif args.phase == "worker":
        if args.tag is None or args.stage is None:
            parser.error("worker requires --tag and --stage")
        evaluate_worker(experiment, args.task, args.tag, args.stage)
    elif args.phase == "swa":
        manifest, _ = load_experiment(experiment)
        ensure_swa(experiment, manifest["entries"][args.task])
    else:
        evaluate(experiment, args.task, args.phase)


if __name__ == "__main__":
    main()
