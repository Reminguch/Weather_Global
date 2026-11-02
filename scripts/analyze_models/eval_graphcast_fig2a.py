#!/usr/bin/env python3
"""Run the public 37-level GraphCast with upstream inference and paper Z500 RMSE."""
from __future__ import annotations

import argparse
import gc as garbage
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.frozen_runtime import verify_source
from scripts.analyze_models.graphcast_fig2a_data import LocalERA5, array_sha, dataset_sha


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def spatial_mse(prediction, truth, weights):
    """Float32 area-weighted spatial MSE, without time pooling or unit changes."""
    prediction = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if prediction.shape != truth.shape or prediction.ndim != 2 or weights.shape != (prediction.shape[0],):
        raise ValueError("Prediction, truth and latitude weights have incompatible shapes")
    if not all(np.isfinite(v).all() for v in (prediction, truth, weights)):
        raise ValueError("Nonfinite prediction/truth/weights")
    error = prediction - truth
    return float(np.mean(error * error * weights[:, None], dtype=np.float32))


def paper_rmse(per_start_mse):
    values = np.asarray(per_start_mse, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Expected finite nonnegative (start, lead) MSE")
    return np.mean(np.sqrt(values), axis=0, dtype=np.float32)


def checked_file(path, receipt, protocol_sha, config_sha):
    if receipt["configuration_sha256"] != config_sha:
        raise ValueError("Wrong staging configuration")
    if Path(receipt["path"]).resolve() != path.resolve():
        raise ValueError("Receipt path differs")
    if path.stat().st_size != receipt["size_bytes"] or sha(path) != receipt["sha256"]:
        raise ValueError(f"Staged file identity changed: {path}")
    return receipt["sha256"]


def evaluate(workflow, shard_index):
    protocol_path = workflow / "protocol.json"
    protocol = read(protocol_path)
    protocol_sha = sha(protocol_path)
    verify_source(Path(protocol["eval_source"]))
    upstream = Path(protocol["upstream_directory"])
    manifest = read(upstream / "manifest.json")
    assert manifest["commit"] == protocol["upstream_commit"]
    for name, digest in manifest["files"].items():
        if sha(upstream / name) != digest:
            raise ValueError(f"Upstream source changed: {name}")
    sys.path.insert(0, str(upstream))
    os.environ["XLA_FLAGS"] = protocol["numerics"]["xla_flags"]
    import haiku as hk
    import jax
    from graphcast import autoregressive, casting, checkpoint, data_utils, losses, normalization, rollout
    from graphcast import graphcast as gc
    if not Path(gc.__file__).resolve().is_relative_to(upstream):
        raise ValueError("Wrong GraphCast implementation imported")
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("This evaluation requires one allocated GPU")
    for identity in [protocol["checkpoint"], *protocol["statistics"]]:
        if sha(identity["path"]) != identity["sha256"]:
            raise ValueError("Checkpoint or normalization identity changed")
    with Path(protocol["checkpoint"]["path"]).open("rb") as stream:
        ckpt = checkpoint.load(stream, gc.CheckPoint)
    task = ckpt.task_config
    if ckpt.model_config.resolution != .25 or list(task.pressure_levels) != protocol["pressure_levels"]:
        raise ValueError("Expected the public 0.25-degree/37-level model")
    if pd.Timedelta(task.input_duration) != pd.Timedelta("12h"):
        raise ValueError("Expected the two-frame input context")
    stats = {Path(i["path"]).stem: xr.load_dataset(i["path"]) for i in protocol["statistics"]}

    def predict(inputs, targets_template, forcings):
        predictor = gc.GraphCast(ckpt.model_config, task)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(predictor, **stats)
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=False)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    transformed = hk.transform_with_state(predict)

    @jax.jit
    def apply(rng, inputs, targets_template, forcings):
        return transformed.apply(ckpt.params, {}, rng, inputs, targets_template, forcings)[0]

    data_root = Path(protocol["data_root"])
    local = None
    if "prepared_root" in protocol:
        local = LocalERA5(protocol["prepared_root"], protocol["prepared_identity"])
    else:
        ready = read(data_root / "READY.json")
        if ready["configuration"]["protocol_sha256"] != protocol_sha:
            raise ValueError("Staging was performed for a different protocol")
        config_sha = ready["configuration_sha256"]
        receipts = {str(Path(r["path"]).resolve()): r for r in ready["files"]}
    if not 0 <= shard_index < protocol["n_shards"]:
        raise ValueError("Shard index out of range")
    selected = protocol["initialization_times"][shard_index::protocol["n_shards"]]
    latitude = np.arange(-90, 90.25, .25, dtype=np.float32)
    longitude = np.arange(0, 360, .25, dtype=np.float32)
    weights = losses.normalized_latitude_weights(xr.DataArray(latitude, dims="lat", coords={"lat": latitude})).values.astype(np.float32)
    begin = time.monotonic()
    result_paths = []
    for initialization in selected:
        stamp = pd.Timestamp(initialization)
        tag = stamp.strftime("%Y%m%dT%H%M%SZ")
        output = workflow / "results" / f"{tag}.json"
        if output.exists():
            old = read(output)
            if old.get("status") != "complete" or old.get("protocol_sha256") != protocol_sha:
                raise ValueError(f"Existing result differs: {output}")
            if sha(old["forecast_path"]) != old["forecast_sha256"]:
                raise ValueError("Existing forecast identity changed")
            result_paths.append(str(output))
            continue
        start_time = time.monotonic()
        if local is not None:
            raw = local.inputs(stamp).sel(level=list(task.pressure_levels))
            input_sha = dataset_sha(raw)
        else:
            input_path = data_root / "inputs" / f"{tag}.nc"
            input_sha = checked_file(input_path, receipts[str(input_path)], protocol_sha, config_sha)
            raw = xr.load_dataset(input_path).sel(level=list(task.pressure_levels))
        np.testing.assert_array_equal(raw.lat, latitude)
        np.testing.assert_array_equal(raw.lon, longitude)
        times = pd.date_range(stamp - pd.Timedelta("6h"), periods=42, freq="6h").tz_localize(None).values
        np.testing.assert_array_equal(raw.time, times[:2])
        for name in raw:
            if not np.isfinite(raw[name].values).all():
                raise ValueError(f"Nonfinite input {name}")
        forcing = xr.Dataset(coords={"time": times, "datetime": ("time", times), "lat": latitude, "lon": longitude})
        data_utils.add_derived_vars(forcing)
        data_utils.add_tisr_var(forcing)
        inputs = xr.merge([raw, forcing[list(task.forcing_variables)].isel(time=slice(0, 2)).drop_vars("datetime")])
        inputs = inputs[list(task.input_variables)].astype(np.float32).assign_coords(
            time=np.array([-6, 0], dtype="timedelta64[h]")).expand_dims(batch=[0])
        future = forcing[list(task.forcing_variables)].isel(time=slice(2, None)).drop_vars("datetime")
        future = future.astype(np.float32).assign_coords(time=np.arange(1, 41) * np.timedelta64(6, "h")).expand_dims(batch=[0]).load()
        del raw, forcing
        one = xr.zeros_like(inputs[list(task.target_variables)].isel(time=slice(-1, None))).assign_coords(
            time=np.array([6], dtype="timedelta64[h]"))
        targets = rollout.extend_targets_template(one, 40)
        scores, predictions, truth_hashes = [], [], []
        for index, prediction in enumerate(rollout.chunked_prediction_generator(
                apply, jax.random.PRNGKey(0), inputs, targets_template=targets,
                forcings=future, num_steps_per_chunk=1)):
            lead = 6 * (index + 1)
            if lead not in protocol["lead_hours"]:
                continue
            np.testing.assert_array_equal(prediction.time.values, [np.timedelta64(lead, "h")])
            z = prediction.geopotential.sel(level=500).isel(batch=0, time=0).transpose("lat", "lon")
            np.testing.assert_array_equal(z.lat, latitude)
            np.testing.assert_array_equal(z.lon, longitude)
            field = np.asarray(z.values, dtype=np.float32)
            valid = stamp + pd.Timedelta(hours=lead)
            if local is not None:
                truth = local.truth(valid)
                truth_hashes.append(array_sha(truth))
                mse = spatial_mse(field, truth, weights)
            else:
                truth_path = data_root / "truth" / f"{valid.strftime('%Y%m%dT%H%M%SZ')}.nc"
                truth_hashes.append(checked_file(truth_path, receipts[str(truth_path)], protocol_sha, config_sha))
                with xr.open_dataset(truth_path) as ds:
                    if pd.Timestamp(ds.attrs["valid_time"]) != valid:
                        raise ValueError("Truth timestamp mismatch")
                    np.testing.assert_array_equal(ds.lat, latitude)
                    np.testing.assert_array_equal(ds.lon, longitude)
                    mse = spatial_mse(field, ds.Z500.values, weights)
            scores.append(mse)
            predictions.append(field.copy())
            print(json.dumps({"initialization": initialization, "lead_hours": lead,
                              "rmse": float(np.sqrt(np.float32(mse))),
                              "elapsed_seconds": time.monotonic() - start_time}), flush=True)
        if len(scores) != 20:
            raise ValueError("Incomplete forecast")
        forecast_path = workflow / "forecasts" / f"{tag}.nc"
        ds = xr.Dataset({"Z500": (("lead_hours", "lat", "lon"), np.stack(predictions))},
                        coords={"lead_hours": protocol["lead_hours"], "lat": latitude, "lon": longitude},
                        attrs={"initialization_time": initialization, "protocol_sha256": protocol_sha,
                               "checkpoint_sha256": protocol["checkpoint"]["sha256"]})
        ds.Z500.attrs["units"] = "m2 s-2"
        temporary = forecast_path.with_suffix(".nc.partial")
        ds.to_netcdf(temporary, engine="netcdf4", encoding={"Z500": {"zlib": True, "complevel": 1}})
        temporary.replace(forecast_path)
        result = {"status": "complete", "protocol_sha256": protocol_sha,
                  "initialization_time": initialization, "lead_hours": protocol["lead_hours"],
                  "mse": scores, "rmse": np.sqrt(np.asarray(scores, dtype=np.float32)).tolist(),
                  "checkpoint_sha256": protocol["checkpoint"]["sha256"],
                  "input_sha256": input_sha, "truth_sha256": truth_hashes,
                  "upstream_commit": manifest["commit"], "graphcast_module": gc.__file__,
                  "forecast_path": str(forecast_path), "forecast_sha256": sha(forecast_path),
                  "elapsed_seconds": time.monotonic() - start_time,
                  "job_id": os.environ.get("SLURM_JOB_ID"), "jax_version": jax.__version__}
        write_json(output, result)
        result_paths.append(str(output))
        del inputs, future, targets, one, predictions, prediction, ds, z, field
        garbage.collect()
    write_json(workflow / "results" / f"shard_{shard_index:02d}.json", {
        "status": "complete", "protocol_sha256": protocol_sha, "results": result_paths,
        "elapsed_seconds": time.monotonic() - begin,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "job_id": os.environ.get("SLURM_JOB_ID"), "gpu": str(jax.devices()[0]),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args()
    evaluate(args.workflow.resolve(), args.shard_index)
