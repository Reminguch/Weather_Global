from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.models.mamba.v24_Ilya import data, evaluation
from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig, parse_args
from src.models.mamba.v24_Ilya.data import EvalDataset, resolve_initialization_manifest
from src.models.mamba.v24_Ilya.metrics import V24IlyaMetricAccumulator


def _manifest(tmp_path, starts, name="starts.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"schema_version": 1, "initializations": starts}))
    return path


def _eval_data():
    times = pd.date_range("2022-12-30", "2024-01-10 18:00", freq="6h")
    return EvalDataset(
        xr.Dataset(coords={"time": times, "lat": [-45.0, 45.0]}),
        pd.Timedelta("6h"),
        2,
    )


def test_full_daily_year_uses_padded_inputs_and_verification_targets(tmp_path):
    starts = pd.date_range("2023-01-01", "2023-12-31", tz="UTC", freq="1D")
    path = _manifest(tmp_path, [value.isoformat() for value in starts])
    records = resolve_initialization_manifest(_eval_data(), path, target_steps=40)
    assert len(records) == 365
    assert records[0]["input_times"] == ["2022-12-31T18:00:00Z", "2023-01-01T00:00:00Z"]
    assert records[-1]["valid_times"][-1] == "2024-01-10T00:00:00Z"
    assert records[-1]["initialization_year"] == 2023
    assert records[-1]["valid_years"][-1] == 2024
    assert records[-1]["lead_hours"] == list(range(6, 241, 6))
    assert records[0]["initialization_id"] == "20230101T000000Z"
    # Three spare six-hour frames suffice; the obsolete 24-step cold
    # reservation would incorrectly exclude these December origins.
    assert _eval_data().dataset.sizes["time"] - records[-1]["dataset_index"] == 44


@pytest.mark.parametrize("missing_time", [
    "2022-12-31T18:00:00", "2023-01-01T00:00:00", "2023-01-10T18:00:00",
])
def test_manifest_rejects_missing_input_origin_or_late_target(tmp_path, missing_time):
    eval_data = _eval_data()
    broken = dataclasses.replace(eval_data, dataset=eval_data.dataset.drop_sel(time=pd.Timestamp(missing_time)))
    path = _manifest(tmp_path, ["2023-01-01T00:00:00Z"])
    with pytest.raises(ValueError, match="missing required input/target"):
        resolve_initialization_manifest(broken, path, target_steps=40)


@pytest.mark.parametrize("entries, message", [
    (["2023-01-01T00:00:00Z"] * 2, "duplicate"),
    (["2023-01-02T00:00:00Z", "2023-01-01T00:00:00Z"], "chronological"),
    (["2023-01-01T00:00:00"], "explicit UTC"),
    (["2023-01-01T00:00:00-05:00"], "must use UTC"),
    (["2023-01-01T00:00:00.500Z"], "must align"),
])
def test_manifest_rejects_ambiguous_dates(tmp_path, entries, message):
    with pytest.raises(ValueError, match=message):
        resolve_initialization_manifest(_eval_data(), _manifest(tmp_path, entries), target_steps=40)


def test_manifest_rejects_duplicate_dataset_dates(tmp_path):
    eval_data = _eval_data()
    broken = dataclasses.replace(eval_data, dataset=xr.concat([eval_data.dataset, eval_data.dataset], dim="time"))
    with pytest.raises(ValueError, match="unique"):
        resolve_initialization_manifest(broken, _manifest(tmp_path, ["2023-01-01T00:00:00Z"]), target_steps=40)


def test_open_manifest_data_retains_year_boundaries_and_legacy_still_slices(tmp_path, monkeypatch):
    original = _eval_data().dataset
    monkeypatch.setattr(data, "open_graphcast_era5", lambda path: original)
    monkeypatch.setattr(data, "select_resolution", lambda ds, resolution: (ds, resolution, 1))
    monkeypatch.setattr(data, "prepare_dataset_for_task", lambda ds, task: ds)
    config = V24IlyaEvalConfig(
        ckpt=tmp_path / "ckpt.pkl", out_json=tmp_path / "result.json",
        eval_mode="cold_full", val_year=2023,
    )
    task = SimpleNamespace(input_duration="12h")
    legacy = data.open_eval_dataset(config, task)
    assert legacy.dataset.sizes["time"] == 1460
    assert config.sample_total_steps == 64
    manifest = _manifest(tmp_path, ["2023-01-01T00:00:00Z", "2023-12-31T00:00:00Z"])
    explicit = dataclasses.replace(config, initialization_manifest=manifest)
    loaded = data.open_eval_dataset(explicit, task)
    assert loaded.dataset.sizes["time"] == 1498
    assert loaded.dataset.time.values[0] == np.datetime64("2022-12-31T18:00:00")
    assert loaded.dataset.time.values[-1] == np.datetime64("2024-01-10T00:00:00")
    assert explicit.sample_total_steps == 40
    assert explicit.total_rollout_steps == 40


def test_development_manifest_bounds_forcing_preparation_after_validating_all_inputs(tmp_path, monkeypatch):
    original = xr.Dataset(coords={
        "time": pd.date_range("2015-01-01", "2022-12-31 18:00", freq="6h"),
        "lat": [-45.0, 45.0],
    })
    source = [original]
    prepared_spans = []
    monkeypatch.setattr(data, "open_graphcast_era5", lambda path: source[0])
    monkeypatch.setattr(data, "select_resolution", lambda ds, resolution: (ds, resolution, 1))

    def prepare(ds, task):
        prepared_spans.append((ds.time.values[0], ds.time.values[-1]))
        return ds

    monkeypatch.setattr(data, "prepare_dataset_for_task", prepare)
    manifest = _manifest(tmp_path, [f"2022-{month:02d}-15T00:00:00Z" for month in (1, 4, 7, 10)])
    config = V24IlyaEvalConfig(
        ckpt=tmp_path / "model.pkl", out_json=tmp_path / "result.json",
        eval_mode="cold_full", initialization_manifest=manifest,
    )
    task = SimpleNamespace(input_duration="12h")
    loaded = data.open_eval_dataset(config, task)
    assert prepared_spans == [(np.datetime64("2022-01-14T18:00"), np.datetime64("2022-10-25T00:00"))]
    records = resolve_initialization_manifest(loaded, manifest, target_steps=40)
    assert len(records) == 4
    assert records[0]["dataset_index"] == 1
    assert records[0]["initialization_id"] == "20220115T000000Z"
    # Missing boundary history must fail before forcing construction, even
    # though a naive date slice would silently shorten the requested range.
    source[0] = original.drop_sel(time=pd.Timestamp("2022-01-14T18:00"))
    with pytest.raises(ValueError, match="missing required input/target"):
        data.open_eval_dataset(config, task)
    assert len(prepared_spans) == 1


def test_manifest_cli_rejects_index_override(tmp_path):
    args = ["--ckpt", "model.pkl", "--out-json", "result.json", "--eval-mode", "cold_full",
            "--initialization-manifest", str(tmp_path / "starts.json"), "--model-id", "M1"]
    assert parse_args(args).model_id == "M1"
    with pytest.raises(ValueError, match="force_idx cannot"):
        parse_args(args + ["--force-idx", "4"])


def _fields(value):
    return xr.Dataset(
        {"2m_temperature": (("batch", "time", "lat", "lon"), np.full((1, 1, 2, 1), value, np.float32))},
        coords={"batch": [0], "time": [np.timedelta64(6, "h")], "lat": [-45.0, 45.0], "lon": [0.0]},
    )


def test_exported_sample_losses_reconstruct_the_existing_exact_aggregate(tmp_path):
    accumulator = V24IlyaMetricAccumulator(
        2, np.asarray([-45.0, 45.0]),
        diffs_stddev_by_level=xr.Dataset({"2m_temperature": xr.DataArray(2.0)}),
        store_spatial_bias=False,
    )
    starts = ["2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z"]
    metadata = resolve_initialization_manifest(_eval_data(), _manifest(tmp_path, starts), target_steps=2)
    records = []
    for sample, entry in enumerate(metadata):
        values = {"baseline": [], "full": []}
        for lead in range(2):
            error = float(2 * (sample + 1) * (lead + 1))
            losses = accumulator.update_step(lead, _fields(0), _fields(error), _fields(error / 2))
            for branch in values:
                values[branch].append(float(losses[branch].item()))
        records.append(evaluation.build_initialization_loss_record(entry, values))
    aggregate = accumulator.finalize()["original_graphcast_loss"]
    for branch in ("baseline", "full"):
        per_start = np.asarray([entry[f"{branch}_per_step"] for entry in records])
        np.testing.assert_allclose(per_start.mean(axis=0), aggregate[f"{branch}_per_step"])
    assert aggregate["improvement_pct_rollout"] == pytest.approx(75)
    with pytest.raises(ValueError, match="Incomplete"):
        evaluation.build_initialization_loss_record(metadata[0], {"baseline": [1, np.nan], "full": [1, 2]})


def test_orchestrator_exports_every_lead_and_keeps_rng_stable_across_shards_and_extension(tmp_path, monkeypatch):
    eval_data = _eval_data()
    starts = ["2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z", "2023-01-03T00:00:00Z"]
    manifest = _manifest(tmp_path, starts)
    checkpoint = SimpleNamespace(task_config=None, model_config=None, params={})
    monkeypatch.setattr(evaluation, "load_graphcast_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr(evaluation, "load_stats", lambda path: {"diffs_stddev_by_level": xr.Dataset({"2m_temperature": xr.DataArray(2.)})})
    monkeypatch.setattr(evaluation, "open_eval_dataset", lambda *args: eval_data)
    predictor = SimpleNamespace(init=lambda *args: ({}, {}), apply=lambda *args: None)
    monkeypatch.setattr(evaluation, "build_predictors", lambda *args: SimpleNamespace(baseline=predictor, residual=predictor))
    monkeypatch.setattr(evaluation, "overlay_frozen_baseline_params", lambda *args: ({}, {}))
    monkeypatch.setattr(evaluation, "load_v24_Ilya_checkpoint", lambda path: SimpleNamespace(residual_params={}, checkpoint_format="test"))
    monkeypatch.setattr(evaluation, "_release_host_memory", lambda: None)
    monkeypatch.setattr(evaluation, "iter_eval_steps", lambda *args, **kwargs: iter(()))

    def build(*args, indices, **kwargs):
        inputs = _fields(0)
        inputs.attrs["index"] = indices[0]
        return inputs, _fields(0), xr.Dataset(coords={"time": [np.timedelta64(6, "h")]})

    seen_keys = {}

    def rollout(*, inputs, rng, prediction_consumer, target_steps, **kwargs):
        idx = inputs.attrs["index"]
        seen_keys[idx] = np.asarray(rng).copy()
        for lead in range(target_steps):
            error = float(idx + lead + 1)
            prediction_consumer(lead, _fields(0), _fields(error), _fields(error / 2))

    monkeypatch.setattr(evaluation, "build_eval_batch", build)
    monkeypatch.setattr(evaluation, "run_v24_Ilya_rollout", rollout)
    config = V24IlyaEvalConfig(
        ckpt=tmp_path / "model.pkl", out_json=tmp_path / "all.json", eval_mode="cold_full",
        initialization_manifest=manifest, model_id="M1", input_duration=None,
        target_steps=2, omit_rms_bias=True,
    )
    observed = []

    class Observer:
        def begin_sample(self, metadata):
            observed.append(("begin", metadata["initialization_time"]))

        def update_step(self, step_index, truth, baseline, full):
            assert isinstance(truth["2m_temperature"].data, np.ndarray)
            np.testing.assert_array_equal(full["2m_temperature"], baseline["2m_temperature"] / 2)
            observed.append(("step", step_index))

        def finish_sample(self):
            observed.append(("finish",))

    complete = evaluation.evaluate_v24_Ilya(config, prediction_observer=Observer())
    assert observed == [event for start in starts for event in
                        [("begin", start), ("step", 0), ("step", 1), ("finish",)]]
    original_keys = dict(seen_keys)
    assert complete["n_samples"] == 3
    assert complete["global_initialization_times"] == starts
    assert complete["sample_total_steps"] == 2
    assert len(complete["sample_elapsed_seconds"]) == 3
    assert all(value >= 0 for value in complete["sample_elapsed_seconds"])
    assert complete["setup_elapsed_seconds"] >= 0
    assert complete["first_sample_elapsed_seconds"] == complete["sample_elapsed_seconds"][0]
    assert complete["steady_sample_elapsed_seconds_mean"] == pytest.approx(np.mean(complete["sample_elapsed_seconds"][1:]))
    records = complete["original_graphcast_loss_per_initialization"]
    assert len(records) == 3
    sharded_records = []
    for shard in range(2):
        result = evaluation.evaluate_v24_Ilya(dataclasses.replace(
            config, out_json=tmp_path / f"shard{shard}.json", anchor_shard_count=2, anchor_shard_index=shard,
        ))
        sharded_records.extend(result["original_graphcast_loss_per_initialization"])
    assert sorted(sharded_records, key=lambda entry: entry["ordinal"]) == records
    short_manifest = _manifest(tmp_path, starts[1:2], "short.json")
    evaluation.evaluate_v24_Ilya(dataclasses.replace(config, initialization_manifest=short_manifest, out_json=tmp_path / "short.json"))
    for idx in original_keys:
        np.testing.assert_array_equal(seen_keys[idx], original_keys[idx])
    timestamp = starts[1]
    assert not np.array_equal(
        evaluation.initialization_rng_key(0, timestamp, stream="rollout"),
        evaluation.initialization_rng_key(0, timestamp, stream="other"),
    )
