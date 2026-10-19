#!/usr/bin/env python3
"""Stream frozen 2023 forecasts onto a shared 1.5-degree verification grid.

M1 and its 1-degree backbone use the completed campaign's frozen evaluator.
The official 0.25-degree backbone uses its unmodified checkpoint architecture,
two observed input frames, generated forcings, and independent feedback. Future
meteorological truth is used only by the common-grid scorer, never as an input.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import xarray as xr


def read_json(path):
    return json.loads(Path(path).read_text())


def path_value(record):
    return Path(record["path"] if isinstance(record, dict) else record).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(record):
    path = path_value(record)
    if isinstance(record, dict) and record.get("sha256"):
        if sha256(path) != record["sha256"]:
            raise ValueError(f"File identity changed: {path}")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def publish_new(path, payload):
    """Publish complete evidence without replacing an existing result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=".common-grid-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def timestamp_id(value):
    return pd.Timestamp(value).strftime("%Y%m%dT%H%M%SZ")


def verify_ready(path, timestamp, kind):
    path = Path(path).resolve()
    receipt = read_json(str(path) + ".READY.json")
    if (receipt.get("schema_version") != 1 or receipt.get("kind") != kind
            or Path(receipt.get("path", "")).resolve() != path
            or pd.Timestamp(receipt.get("timestamp")) != pd.Timestamp(timestamp)
            or receipt.get("size_bytes") != path.stat().st_size):
        raise ValueError(f"Staging receipt does not describe the requested data: {path}")
    digest = sha256(path)
    if receipt.get("sha256") != digest:
        raise ValueError(f"Staged file changed after validation: {path}")
    return receipt


def records_from_manifest(manifest, target_steps=40):
    body = read_json(manifest)
    if body.get("schema_version") != 1 or not body.get("initializations"):
        raise ValueError("Expected a nonempty schema-1 initialization manifest")
    result = []
    for entry in body["initializations"]:
        value = entry["initialization_time"] if isinstance(entry, dict) else entry
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None or stamp.utcoffset() != pd.Timedelta(0):
            raise ValueError("Explicit UTC initialization timestamps are required")
        if stamp != stamp.floor("6h"):
            raise ValueError("Initialization does not align to a six-hour boundary")
        valid = pd.date_range(stamp + pd.Timedelta("6h"), periods=target_steps, freq="6h")
        result.append({
            "initialization_time": stamp.isoformat().replace("+00:00", "Z"),
            "valid_times": [v.isoformat().replace("+00:00", "Z") for v in valid],
            "lead_hours": [6 * (i + 1) for i in range(target_steps)],
        })
    times = [r["initialization_time"] for r in result]
    if times != sorted(set(times)):
        raise ValueError("Initialization times must be unique and chronological")
    return result


def load_helper():
    # Frozen historical scripts namespaces do not contain this new helper.
    path = Path(__file__).with_name("common_grid_metrics.py")
    spec = importlib.util.spec_from_file_location("common_grid_2023_metrics", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_comparison_protocol(protocol, protocol_path):
    body = {k: v for k, v in protocol.items() if k != "protocol_hash"}
    calculated = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if protocol.get("protocol_hash") != calculated:
        raise ValueError("Common-grid protocol self-hash does not match")
    verify_file({"path": path_value(protocol["original_campaign"]) / "protocol.json",
                 "sha256": protocol["original_protocol_sha256"]})
    verify_file({"path": protocol["initialization_manifest"],
                 "sha256": protocol["initialization_manifest_sha256"]})
    source = path_value(protocol["source_root"])
    snapshot = read_json(source / "snapshot.json")
    files = snapshot["files"]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if snapshot.get("source_digest") != digest or protocol["source_digest"] != digest:
        raise ValueError("Common-grid source snapshot identity differs from protocol")
    for relative, expected in files.items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source):
            raise ValueError("Source path escapes frozen snapshot")
        verify_file({"path": path, "sha256": expected})
    if Path(__file__).resolve() != source / "scripts/analyze_models/eval_common_grid_2023.py":
        raise ValueError("Run the evaluator from the frozen comparison source")
    for record in protocol["statistics"]:
        verify_file(record)


def configure_frozen_runtime(protocol):
    campaign = path_value(protocol["original_campaign"])
    original = read_json(campaign / "protocol.json")
    original_body = {k: v for k, v in original.items() if k != "protocol_hash"}
    calculated = hashlib.sha256(json.dumps(original_body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if calculated != original.get("protocol_hash"):
        raise ValueError("Historical protocol self-hash does not match")
    expected_original = protocol.get("original_protocol_hash")
    if expected_original is not None and original.get("protocol_hash") != expected_original:
        raise ValueError("The reference campaign differs from the frozen comparison protocol")
    source = Path(original["source_root"])
    if not source.is_dir():
        raise FileNotFoundError(source)
    # Do this before importing any GraphCast or project implementation modules.
    sys.path[:0] = [str(source), str(source / "third_party/graphcast")]
    required = {"XLA_FLAGS": original["numerics"]["xla_flags"],
                **original["numerics"].get("environment", {})}
    for key, value in required.items():
        if key in os.environ and os.environ[key] != value:
            raise ValueError(f"{key} differs from the selected model's frozen runtime")
        os.environ[key] = value
    for package, version in original["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Package version differs from the frozen campaign: {package}")
    for relative, digest in read_json(verify_file(original["source_hashes"])).items():
        verify_file({"path": source / relative, "sha256": digest})
    return original


class CommonTruth:
    def __init__(self, path):
        self.path = Path(path)
        self.dataset = None
        if self.path.is_file():
            self.dataset = xr.open_dataset(self.path)
        elif self.path.suffix == ".zarr":
            raise ValueError("Use staged per-valid-time NetCDF truth with readiness receipts")

    def at(self, valid_time):
        expected = pd.Timestamp(valid_time).tz_convert("UTC").tz_localize(None).to_datetime64()
        if self.dataset is not None:
            return self.dataset.sel(time=expected).load(), sha256(self.path)
        path = self.path / (timestamp_id(valid_time) + ".nc")
        receipt = verify_ready(path, valid_time, "common_truth")
        with xr.open_dataset(path) as source:
            result = source.load()
        if "time" in result.coords:
            actual = np.asarray(result.time.values).reshape(-1).astype("datetime64[ns]")
            if len(actual) != 1 or actual[0] != expected:
                raise ValueError(f"Truth timestamp differs from filename: {path}")
            if "time" in result.dims:
                result = result.isel(time=0, drop=True)
        if "valid_time" in result.attrs and pd.Timestamp(result.attrs["valid_time"]) != pd.Timestamp(valid_time):
            raise ValueError(f"Truth valid-time attribute differs from receipt: {path}")
        return result, receipt["sha256"]


class CommonGridObserver:
    def __init__(self, truth, helper, target_steps):
        self.truth = truth
        self.helper = helper
        self.target_steps = target_steps
        self.records = []
        self.current = None

    def begin_sample(self, metadata):
        if self.current is not None:
            raise ValueError("Previous initialization did not finish")
        self.current = {k: metadata[k] for k in
                        ("initialization_time", "valid_times", "lead_hours")}
        if len(self.current["valid_times"]) != self.target_steps:
            raise ValueError("Unexpected common-grid forecast horizon")
        self.current["mse"] = {}
        self.current["truth_sha256"] = []

    def score(self, step_index, predictions):
        if self.current is None or not 0 <= step_index < self.target_steps:
            raise ValueError("Invalid sample or lead")
        truth, digest = self.truth.at(self.current["valid_times"][step_index])
        if len(self.current["truth_sha256"]) != step_index:
            raise ValueError("Duplicate or nonsequential truth verification")
        self.current["truth_sha256"].append(digest)
        for branch, prediction in predictions.items():
            values = self.helper.score_step(prediction, truth)
            scored = self.current["mse"].setdefault(branch, {})
            for diagnostic, value in values.items():
                series = scored.setdefault(diagnostic, [])
                if len(series) != step_index or not np.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid or duplicate score: {branch}/{diagnostic}")
                series.append(float(value))

    def update_step(self, index, native_truth, baseline, full):
        # native_truth is independently scored by the frozen evaluator; every
        # model in this comparison uses the same external common-grid truth.
        self.score(index, {"gc1": baseline, "m1": full})

    def finish_sample(self):
        if self.current is None or not self.current["mse"]:
            raise ValueError("Cannot finish an empty initialization")
        for branch in self.current["mse"].values():
            if len(branch) != 12 or any(len(v) != self.target_steps for v in branch.values()):
                raise ValueError("Incomplete common-grid initialization")
        self.records.append(self.current)
        self.current = None


def run_m1(args, protocol, original, observer, manifest):
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya

    model = next(m for m in original["models"] if m["model_id"] == "M1")
    for path, digest in ((model["checkpoint"], model["checkpoint_sha256"]),
                         (original["baseline"]["path"], original["baseline"]["checkpoint_sha256"])):
        verify_file({"path": path, "sha256": digest})
    for record in original["statistics"]:
        verify_file(record)
    native = args.output_dir / "native" / f"m1_{args.shard_index:02d}.json"
    if native.exists():
        raise FileExistsError(f"Native output already exists without completed common shard: {native}")
    config = V24IlyaEvalConfig(
        ckpt=Path(model["checkpoint"]), out_json=native,
        ckpt_in=Path(original["baseline"]["path"]), stats_dir=Path(original["stats_dir"]),
        data_path=original["data_path"], val_year=2023,
        train_start_year=2015, train_end_year=2021,
        target_steps=protocol.get("target_steps", 40),
        n_samples=len(records_from_manifest(manifest, protocol.get("target_steps", 40))),
        omit_rms_bias=True, initialization_manifest=manifest, model_id="M1",
        anchor_shard_count=args.shard_count, anchor_shard_index=args.shard_index,
        reset_state_every_step=False, **original["forecast"], **model["architecture"],
    )
    native.parent.mkdir(parents=True, exist_ok=True)
    result = evaluate_v24_Ilya(config, prediction_observer=observer)
    return {"native_result": str(native), "native_result_sha256": sha256(native),
            "original_graphcast_loss": result["original_graphcast_loss"],
            "sample_elapsed_seconds": result["sample_elapsed_seconds"],
            "setup_elapsed_seconds": result.get("setup_elapsed_seconds"),
            "checkpoint_sha256": model["checkpoint_sha256"],
            "baseline_checkpoint_sha256": original["baseline"]["checkpoint_sha256"]}


def prepare_native_inputs(path, record, task):
    from src.models.graphcast.training.core.dataset import prepare_dataset_for_task

    with xr.open_dataset(path) as source:
        data = source.load()
    if "batch" in data.dims:
        if data.sizes["batch"] != 1:
            raise ValueError("Expected one initialization in the staged input file")
        data = data.isel(batch=0, drop=True)
    expected = pd.date_range(pd.Timestamp(record["initialization_time"]) - pd.Timedelta("6h"),
                             periods=2, freq="6h").tz_localize(None).values
    if not np.array_equal(data.time.values.astype("datetime64[ns]"), expected):
        raise ValueError(f"Staged input times disagree with manifest: {path}")
    for coord, grid in (("lat", np.arange(-90., 90.01, .25)),
                        ("lon", np.arange(0., 360., .25))):
        if data[coord].shape != grid.shape or not np.allclose(data[coord], grid, rtol=0, atol=1e-6):
            raise ValueError(f"Unexpected native 0.25-degree {coord} coordinates")
    data = data.sel(level=list(task.pressure_levels))
    data = prepare_dataset_for_task(data, task)
    data = data[list(task.input_variables)].drop_vars("datetime", errors="ignore")
    for name, variable in data.data_vars.items():
        if not np.isfinite(variable.values).all():
            raise ValueError(f"Nonfinite native input: {name}")
    data = data.astype(np.float32).assign_coords(time=np.array([-6, 0], dtype="timedelta64[h]"))
    return data.expand_dims(batch=[0])


def native_forcings(inputs, valid_time, lead_hours, task):
    from graphcast import data_utils

    stamp = pd.Timestamp(valid_time).tz_convert("UTC").tz_localize(None).to_datetime64()
    forcing = xr.Dataset(coords={
        "time": np.array([stamp], dtype="datetime64[ns]"),
        "datetime": ("time", np.array([stamp], dtype="datetime64[ns]")),
        "lat": inputs.lat.values, "lon": inputs.lon.values,
    })
    data_utils.add_derived_vars(forcing)
    if "toa_incident_solar_radiation" in task.forcing_variables:
        data_utils.add_tisr_var(forcing)
    missing = set(task.forcing_variables) - set(forcing.data_vars)
    if missing:
        raise ValueError(f"Cannot generate required future forcing: {sorted(missing)}")
    return (forcing[list(task.forcing_variables)].drop_vars("datetime", errors="ignore")
            .astype(np.float32).assign_coords(time=np.array([lead_hours], dtype="timedelta64[h]"))
            .expand_dims(batch=[0]).load())


def native_target_template(inputs, task):
    # Values are ignored by GraphCast prediction; truth remains only in scorer.
    return xr.zeros_like(inputs[list(task.target_variables)].isel(time=slice(-1, None)))


def run_gc025(args, protocol, observer, records):
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from src.models.graphcast.training.core.model import load_graphcast_checkpoint, load_stats
    from src.models.mamba.v24_Ilya.checkpoint import overlay_frozen_baseline_params
    from src.models.mamba.v24_Ilya.evaluation import initialization_rng_key, _release_host_memory
    from src.models.mamba.v24_Ilya.model import make_baseline_predictor
    from src.models.mamba.v24_Ilya.rollout import physical_fields_fp32, shift_inputs_with_field, state_boundary_fp32

    checkpoint_path = verify_file({"path": path_value(protocol["gc025_checkpoint"]),
                                   "sha256": protocol["gc025_checkpoint_sha256"]})
    checkpoint = load_graphcast_checkpoint(checkpoint_path)
    task = checkpoint.task_config
    if checkpoint.model_config.resolution != .25 or len(task.pressure_levels) != 37:
        raise ValueError("Expected the official 0.25-degree, 37-level GraphCast checkpoint")
    if pd.Timedelta(task.input_duration) != pd.Timedelta("12h"):
        raise ValueError("Expected two six-hour input frames")
    stats = load_stats(path_value(protocol.get("stats_dir", "data/graphcast/graphcast/stats")))

    def predict(inputs, template, forcings):
        return make_baseline_predictor(checkpoint.model_config, task, stats, use_bf16=True)(
            inputs, targets_template=template, forcings=forcings)

    transformed = hk.transform_with_state(predict)
    step = jax.jit(transformed.apply)
    @jax.jit
    def fields_finite(fields):
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree_util.tree_leaves(fields)]))
    params, initial_state = None, None
    inputs_dir = path_value(protocol["inputs025"])
    input_records, durations = [], []
    for record in records:
        started = time.monotonic()
        input_path = inputs_dir / (timestamp_id(record["initialization_time"]) + ".nc")
        input_receipt = verify_ready(input_path, record["initialization_time"], "inputs025")
        current = prepare_native_inputs(input_path, record, task)
        template = native_target_template(current, task)
        first_forcing = native_forcings(current, record["valid_times"][0], 6, task)
        if params is None:
            initialized, initial_state = transformed.init(
                jax.random.PRNGKey(0), current,
                template.assign_coords(time=first_forcing.time), first_forcing)
            params, _ = overlay_frozen_baseline_params(initialized, checkpoint.params)
            initial_state = state_boundary_fp32(initial_state)
            del initialized
        state = initial_state
        rng = initialization_rng_key(0, record["initialization_time"], stream="rollout")
        observer.begin_sample(record)
        for index, valid_time in enumerate(record["valid_times"]):
            lead_started = time.monotonic()
            forcing = first_forcing if index == 0 else native_forcings(current, valid_time, 6 * (index + 1), task)
            target = template.assign_coords(time=forcing.time)
            rng, baseline_key, _ = jax.random.split(rng, 3)
            prediction, state = step(params, state, baseline_key, current, target, forcing)
            prediction = physical_fields_fp32(prediction)
            state = state_boundary_fp32(state)
            if not bool(fields_finite(prediction)):
                raise ValueError(f"Nonfinite GraphCast output at {valid_time}")
            observer.score(index, {"gc025": prediction})
            if len(observer.records) == 0 and index < 3:
                print(f"[common-grid] gc025 lead={6 * (index + 1)}h "
                      f"elapsed={time.monotonic() - lead_started:.2f}s", flush=True)
            if index + 1 < len(record["valid_times"]):
                current = shift_inputs_with_field(current, prediction, forcing,
                                                   time_step=pd.Timedelta("6h"), input_steps=2)
            del prediction, forcing, target
        observer.finish_sample()
        input_records.append(input_receipt)
        durations.append(time.monotonic() - started)
        print(f"[common-grid] gc025 {record['initialization_time']} {durations[-1]:.1f}s", flush=True)
        del current, template, first_forcing
        _release_host_memory()
    return {"checkpoint_sha256": sha256(checkpoint_path), "inputs": input_records,
            "sample_elapsed_seconds": durations,
            "numerical_contract": "native_graphcast_bfloat16_with_float32_physical_fields"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=("m1", "gc025"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--initializations", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("Require 0 <= shard-index < shard-count")
    args.workflow = args.workflow.resolve()
    args.output_dir = (args.output_dir or args.workflow / "shards").resolve()
    protocol_path = args.workflow / "protocol.json"
    protocol = read_json(protocol_path)
    validate_comparison_protocol(protocol, protocol_path)
    original = configure_frozen_runtime(protocol)
    import jax
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("Forecast evaluation requires exactly one allocated GPU")
    manifest = args.initializations or path_value(protocol.get("initialization_manifest", original["initialization_manifest"]))
    records = records_from_manifest(manifest, protocol.get("target_steps", 40))
    selected = records[args.shard_index::args.shard_count]
    if not selected:
        raise ValueError("Empty evaluation shard")
    out = args.output_dir / f"{args.model}_{args.shard_index:02d}.json"
    if out.exists():
        raise FileExistsError(f"Completed result already exists: {out}")
    helper = load_helper()
    observer = CommonGridObserver(CommonTruth(path_value(protocol["common_truth"])), helper,
                                  protocol.get("target_steps", 40))
    started = time.monotonic()
    if args.model == "m1":
        evidence = run_m1(args, protocol, original, observer, manifest)
    else:
        evidence = run_gc025(args, protocol, observer, selected)
    if [r["initialization_time"] for r in observer.records] != [r["initialization_time"] for r in selected]:
        raise ValueError("Output initialization coverage differs from the requested shard")
    publish_new(out, {
        "schema_version": 1, "status": "complete", "model": args.model,
        "protocol_hash": protocol.get("protocol_hash", sha256(protocol_path)),
        "protocol_sha256": sha256(protocol_path), "original_protocol_hash": original["protocol_hash"],
        "initialization_manifest_sha256": sha256(manifest),
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "per_initialization": observer.records, "evidence": evidence,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "devices": [str(device) for device in jax.devices()],
        "gpu_kind": jax.devices()[0].device_kind,
    })
    print(f"[common-grid] completed {out}", flush=True)


if __name__ == "__main__":
    main()
