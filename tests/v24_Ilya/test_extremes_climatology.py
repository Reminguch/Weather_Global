"""Scientific boundary checks for independently fitted extreme thresholds."""
import json

import numpy as np
import pytest

from src.analysis.extremes_climatology import (
    QUANTILES,
    calendar_day_index,
    calendar_day_indices,
    circular_window_mask,
    compute_quantiles,
    reference_indices,
    sha256,
    tp24_mm,
)


def test_calendar_handles_leap_year_without_march_shift():
    times = np.array(["2019-02-28", "2020-02-28", "2020-02-29", "2020-03-01",
                      "2019-03-01", "2024-01-01", "2024-12-31"], dtype="datetime64[D]")
    assert calendar_day_indices(times).tolist() == [58, 58, -1, 59, 59, 0, 364]
    assert calendar_day_index("2024-01-10T18:00:00") == 9
    with pytest.raises(ValueError, match="February 29"):
        calendar_day_index("2024-02-29T00:00:00")


def test_calendar_window_wraps_year_and_excludes_feb29():
    days = np.array([-1, 0, 30, 31, 333, 334, 335, 364])
    assert circular_window_mask(days, 0, 30).tolist() == [False, True, True, False, False, False, True, True]
    assert np.count_nonzero(circular_window_mask(np.arange(365), 0, 30)) == 61


def test_precipitation_uses_raw_antecedents_across_boundaries():
    # Jan1 00UTC accumulates Dec31 06/12/18UTC and Jan1 00UTC. Negative
    # raw values are retained; clipping would change this known total.
    intervals = np.array([0.001, -0.0005, 0.002, 0.003], dtype=np.float32)[:, None, None]
    assert float(tp24_mm(intervals)[0, 0]) == pytest.approx(5.5)
    times = np.arange(np.datetime64("2019-12-31T00"), np.datetime64("2021-01-01T00"),
                      np.timedelta64(6, "h"))
    indices = reference_indices(times, 0, 2020, 2020)
    assert len(indices) == 365
    march_index = indices[np.flatnonzero(times[indices] == np.datetime64("2020-03-01T00"))[0]]
    assert times[march_index - 3] == np.datetime64("2020-02-29T06")
    assert not np.any(times[indices].astype("datetime64[D]") == np.datetime64("2020-02-29"))
    with pytest.raises(ValueError, match="four finite"):
        tp24_mm(np.array([0.0, 1.0, np.nan, 2.0])[:, None, None])


def test_tiled_exact_quantiles_and_completion_integrity(tmp_path):
    # Two years with a known cyclic seasonal signal and different offsets;
    # calculate a January boundary example independently as a 122-value set.
    dates = np.arange(np.datetime64("2019-01-01"), np.datetime64("2021-01-01"))
    dates = dates[calendar_day_indices(dates) >= 0]
    day = calendar_day_indices(dates)
    offset = (dates.astype("datetime64[Y]").astype(int) - 49) * 1000
    base = day + offset
    fields = np.stack([base, -base, base * 2, base + 7, base / 3, base * 0], axis=-1).reshape(730, 2, 3).astype(np.float32)
    stage = tmp_path / "reference.npy"
    np.save(stage, fields)
    metadata = dict(shape=list(fields.shape), sha256=sha256(stage), signature=dict(
        observable="T2m", hour=0, start_year=2019, end_year=2020,
        reference_dates=[str(v) for v in dates]))
    record = compute_quantiles(stage, metadata, tmp_path, tile_points=2)
    values = np.load(tmp_path / record["path"])
    # For Jan1 the circular reference includes days335..364 and0..30.
    hand_days = np.concatenate([np.arange(335, 365), np.arange(31)])
    hand_samples = np.concatenate([hand_days, hand_days + 1000])
    expected = np.quantile(hand_samples, QUANTILES, method="linear")
    np.testing.assert_allclose(values[0, :, 0, 0], expected)
    assert record["sample_counts"] == [122] * 365
    assert json.loads((tmp_path / "T2m_h00.json").read_text())["sha256"] == sha256(tmp_path / "T2m_h00.npy")
    assert compute_quantiles(stage, metadata, tmp_path, tile_points=5)["sha256"] == record["sha256"]
    # Existing output is not silently accepted if bytes differ.
    damaged = np.load(tmp_path / "T2m_h00.npy", mmap_mode="r+")
    damaged[0, 0, 0, 0] += 1
    damaged.flush()
    with pytest.raises(ValueError, match="integrity"):
        compute_quantiles(stage, metadata, tmp_path, tile_points=2)
