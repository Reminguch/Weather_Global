from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest
import xarray as xr

from src.analysis.extremes_observer import (
    ExtremesObserver, STATISTICS, calendar_day_index, factory, occurrence_statistics,
    sha256, spherical_cell_weights,
)


LAT = np.asarray([-90., 0., 90.])
LON = np.asarray([0., 90., 180., 270.])


def test_cell_areas_include_polar_caps_and_reverse_consistently():
    area = spherical_cell_weights(LAT, LON)
    assert area.sum() == pytest.approx(1)
    assert np.all(area > 0)
    np.testing.assert_allclose(area, spherical_cell_weights(LAT[::-1], LON)[::-1])
    assert calendar_day_index(datetime(2024, 1, 9)) == 8
    assert calendar_day_index(datetime(2023, 12, 31)) == 364


def test_false_alarms_and_conditional_errors_remain_distinct():
    truth = np.asarray([0., 3., 3., 0.])
    pred = np.asarray([4., 4., 0., 0.])
    result = dict(zip(STATISTICS, occurrence_statistics(truth, pred, np.ones(4),
                                                       np.ones(4, bool), np.full(4, .25))))
    assert [result[x] for x in STATISTICS[:4]] == [.25] * 4
    assert [result[x] for x in STATISTICS[5:9]] == [1] * 4
    assert result["tail_absolute_error_sum"] == 1
    assert result["tail_signed_error_sum"] == -.5
    assert result["tail_squared_error_sum"] == 2.5
    # A false alarm remains counted even when there are no observed extremes.
    result = occurrence_statistics(np.zeros(4), pred, np.ones(4), np.ones(4, bool), np.full(4, .25))
    assert result[2] == .5
    assert np.all(result[10:] == 0)


def test_empty_eligible_domain_has_zero_sums_not_invented_skill():
    result = occurrence_statistics(np.ones(4), np.ones(4), np.zeros(4), np.zeros(4, bool), np.full(4, .25))
    assert np.all(result == 0)
    assert result[0] + result[1] + result[2] == 0  # CSI denominator undefined.


def test_cold_inequality_and_threshold_ties_are_explicit():
    result = occurrence_statistics(np.asarray([-1., 0., 1., 0.]), np.asarray([-2., -1., 1., 0.]),
                                   np.zeros(4), np.ones(4, bool), np.full(4, .25), cold=True)
    assert result[0] == .25
    assert result[2] == .25  # Truth equal to threshold is not an observed cold event.
    assert result[11] == -.25


class Climatology:
    latitude, longitude = LAT, LON

    def thresholds(self, observable, _time):
        values = {"T2m": [270, 275, 295, 300], "WS10": [0, 0, 8, 10], "TP24": [0, 0, 2, 4]}[observable]
        return np.broadcast_to(np.asarray(values, float)[:, None, None], (4, 3, 4)).copy()


def fields(t2m=301., ws10=12., rain=0.001):
    values = {"2m_temperature": t2m, "10m_u_component_of_wind": ws10,
              "10m_v_component_of_wind": 0., "total_precipitation_6hr": rain}
    return xr.Dataset({name: (("lat", "lon"), np.full((3, 4), value)) for name, value in values.items()},
                      coords={"lat": LAT, "lon": LON})


def metadata(steps):
    initial = datetime(2023, 12, 30, tzinfo=timezone.utc)
    return dict(initialization_id="20231230T000000Z", initialization_time=initial.isoformat(),
                lead_hours=list(range(6, steps * 6 + 1, 6)),
                valid_times=[(initial + timedelta(hours=6 * (i + 1))).isoformat() for i in range(steps)])


def observer(tmp_path, steps=5):
    return ExtremesObserver(tmp_path, {key: key for key in
                            ("model_id", "protocol_hash", "checkpoint_sha256", "baseline_checkpoint_sha256")},
                            Climatology(), np.ones((3, 4)), target_steps=steps)


def test_streaming_tp24_uses_four_forecast_steps_and_retains_negative_rain(tmp_path):
    writer = observer(tmp_path)
    writer.begin_sample(metadata(5))
    truth_rain = [.001, .001, .001, .003, .002]
    pred_rain = [.001, .001, .001, -.001, .002]
    for step in range(5):
        truth, pred = fields(rain=truth_rain[step]), fields(rain=pred_rain[step])
        writer.update_step(step, truth, truth, pred)
    writer.finish_sample()
    path = tmp_path / "extremes/20231230T000000Z.npz"
    with np.load(path, allow_pickle=False) as record:
        assert record["stats"].shape == (5, 4, 2, 2, 2, 15)
        assert np.isnan(record["stats"][:3, 3]).all()
        assert not record["valid_step_event"][:3, 3].any()
        # Day1 truth=6mm; full=2mm. A later window drops the first 1mm.
        assert record["stats"][3, 3, 0, 0, 1, 1] == pytest.approx(1)
        assert record["stats"][3, 3, 0, 0, 1, 10] == pytest.approx(4)
        assert record["stats"][4, 3, 0, 0, 1, 14] == pytest.approx(3)
        assert record["negative_tp6_counts"][3, 2] == 12
        # Identical baseline/truth has only hits and correct negatives and zero errors.
        assert np.all(record["stats"][:, :3, :, :, 0, 1:3] == 0)
        assert np.all(record["stats"][:, :3, :, :, 0, 10:13] == 0)
        saved = json.loads(str(record["metadata"]))
        assert saved["status"] == "complete"
    with pytest.raises(FileExistsError):
        writer.begin_sample(metadata(5))


def test_integrity_failures_do_not_publish_partial_results(tmp_path):
    writer = observer(tmp_path, steps=2)
    writer.begin_sample(metadata(2))
    with pytest.raises(ValueError, match="out-of-order"):
        writer.update_step(1, fields(), fields(), fields())
    writer.update_step(0, fields(), fields(), fields())
    with pytest.raises(ValueError, match="incomplete"):
        writer.finish_sample()
    with pytest.raises(ValueError, match="grids differ"):
        writer.update_step(1, fields(), fields().assign_coords(lon=LON + 1), fields())
    assert not list(tmp_path.rglob("*.npz"))


def test_factory_loads_pinned_climatology_and_rejects_tampering(tmp_path):
    from scripts.experiments.run_v24_spectral_pilot import protocol_digest

    climate = tmp_path / "climatology"
    climate.mkdir()
    arrays = []
    for observable in ("T2m", "WS10", "TP24"):
        for hour in (0, 6, 12, 18):
            path = climate / f"{observable}_h{hour:02d}.npy"
            values = Climatology().thresholds(observable, None)
            np.save(path, np.broadcast_to(values, (365, *values.shape)).astype(np.float32))
            arrays.append(dict(path=path.name, sha256=sha256(path), observable=observable, hour=hour))
    for name, values in (("lat", LAT), ("lon", LON)):
        path = climate / f"{name}.npy"
        np.save(path, values)
        arrays.append(dict(path=path.name, sha256=sha256(path), coordinate=name))
    climate_meta = climate / "metadata.json"
    climate_meta.write_text(json.dumps(dict(version=1, status="complete", arrays=arrays,
                                           quantiles=[.02, .05, .95, .98])))
    land_path = tmp_path / "land_mask.npy"
    np.save(land_path, np.ones((3, 4), dtype=np.float32))
    pinned = lambda path: dict(path=str(path), sha256=sha256(path))
    manifest = dict(format="v24_amse_regime_evaluation_v1", schema_version=1,
                    extremes=dict(climatology_dir=str(climate), climatology_metadata=pinned(climate_meta),
                                  land_mask=pinned(land_path), protocol_id="synthetic-factory-test"))
    manifest["manifest_hash"] = protocol_digest(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    spec = dict(experiment_manifest=pinned(manifest_path), target_steps=1)
    identity = {key: key for key in
                ("model_id", "protocol_hash", "checkpoint_sha256", "baseline_checkpoint_sha256")}
    writer = factory(spec, tmp_path / "result", identity)
    writer.begin_sample(metadata(1))
    writer.update_step(0, fields(), fields(), fields())
    writer.finish_sample()
    with np.load(tmp_path / "result/extremes/20231230T000000Z.npz", allow_pickle=False) as record:
        saved = json.loads(str(record["metadata"]))
        assert saved["provenance"]["extremes_protocol"]["climatology_metadata"] == pinned(climate_meta)
        assert record["valid_step_event"].tolist() == [[True, True, True, False]]
        assert np.all(record["stats"][0, :3, :, :, :, 10:13] == 0)
    # Extra bytes are ignored by NumPy's array decoder but must fail provenance.
    with (climate / "T2m_h00.npy").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        factory(spec, tmp_path / "tampered-result", identity)
