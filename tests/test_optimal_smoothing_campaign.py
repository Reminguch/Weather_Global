import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from scripts.experiments import run_v24_optimal_smoothing as campaign
from scripts.experiments.run_v24_mgda import unseal


def cohorts():
    first = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return ({"initializations": [f"2022-01-{day:02d}T00:00:00Z" for day in range(1, 32)]
                                + ["2022-02-01T00:00:00Z"]},
            {"initializations": [(first + timedelta(days=3 * i)).isoformat() for i in range(122)]})


def test_fit_year_and_exact_v4_order_are_required():
    development, annual = cohorts()
    campaign.validate_cohorts(development, annual)
    with pytest.raises(ValueError, match="2022"):
        campaign.validate_cohorts({"initializations": annual["initializations"][:32]}, annual)
    reversed_annual = {"initializations": list(reversed(annual["initializations"]))}
    with pytest.raises(ValueError, match="ordered"):
        campaign.validate_cohorts(development, reversed_annual)
    wrong_hour = json.loads(json.dumps(annual))
    wrong_hour["initializations"][0] = "2023-01-01T06:00:00Z"
    with pytest.raises(ValueError, match="ordered"):
        campaign.validate_cohorts(development, wrong_hour)


def test_fitter_pools_dates_before_ratio_and_preserves_all_leads(tmp_path, monkeypatch):
    paths = [tmp_path / "20220101T000000Z.npz", tmp_path / "20220201T000000Z.npz"]
    for path in paths:
        path.write_bytes(b"fixture-pinned-statistics")
    monkeypatch.setattr(campaign.original, "statistics_paths", lambda *args: paths)
    shape = (2, 40, 2, 3)
    layout = {"channels": [["u", 850], ["v", 850]], "group_index": [0, 0]}

    def load(path, manifest, stage):
        assert stage == "calibrate"
        p = np.ones(shape) * (1 if path == paths[0] else 4)
        c = np.ones(shape) * (1 if path == paths[0] else 2)
        c[:, 1, 0] *= -2
        c[:, 2, 1] *= 3
        return {"prediction_power": p, "cross_power": c}, {
            "branches": ["GC", "M1"], "lead_hours": list(range(6, 241, 6)), "layout": layout}

    monkeypatch.setattr(campaign.original, "load_statistics", load)
    campaign.fit(tmp_path, {"record_hash": "campaign", "model_id": "M1",
                           "policy": {"lmax": 2, "operator": "S[b A f]"}})
    frozen = unseal(tmp_path / "filters.json")
    gains = np.asarray(frozen["branches"]["GC"]["gains"])
    np.testing.assert_allclose(gains[0], .6)
    np.testing.assert_allclose(gains[1, 0], -1.2)
    np.testing.assert_allclose(gains[1, 1], .6)
    np.testing.assert_allclose(gains[2, 1], 1.8)
    assert frozen["development_count"] == 2 and frozen["fit_year"] == 2022
    assert gains.shape == (40, 2, 3)


def test_submission_orders_calibration_fit_evaluation_report(tmp_path, monkeypatch):
    calls = []

    def submit_once(root, name, command):
        calls.append((name, command))
        return {"job_id": str(100 + len(calls))}

    monkeypatch.setattr(campaign, "submit_once", submit_once)
    manifest = {"workspace": str(tmp_path), "source_root": str(tmp_path / "source"), "model_id": "M1",
        "shards": {"calibrate": [{}] * 4, "evaluate": [{}] * 8},
        "resource_measurement": {"memory_gib": 20, "calibration_minutes": 62, "evaluation_minutes": 90}}
    campaign.submit(tmp_path, manifest)
    assert [name for name, _ in calls] == ["calibrate", "fit", "evaluate", "report"]
    for index, (_, command) in enumerate(calls):
        assert command[-5].endswith(campaign.WRAPPER)
        if index:
            assert f"--dependency=afterok:{100 + index}" in command
        assert not any("%" in arg for arg in command if arg.startswith("--array="))
    assert "--array=0-3" in calls[0][1] and "--array=0-7" in calls[2][1]
