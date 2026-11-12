"""CPU checks for the small raw-checkpoint comparison and its saved evidence."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments import run_v24_raw_checkpoint_pilot as pilot


def test_quarter_subset_is_score_independent_and_preserves_source_indices():
    stamps = ["2022-08-06T12:00:00Z", "2022-04-16T12:00:00Z",
              "2022-01-06T18:00:00Z", "2022-10-07T06:00:00Z",
              "2022-07-07T18:00:00Z", "2022-04-03T06:00:00Z",
              "2022-01-02T00:00:00Z", "2022-12-01T12:00:00Z"]
    development = {"initializations": [{"initialization_time": stamp} for stamp in stamps],
                   "source_chosen_idx": [870, 422, 23, 1117, 751, 369, 4, 1338]}
    before = deepcopy(development)
    result = pilot.pilot_cohort(development)
    assert development == before
    assert [row["initialization_time"] for row in result["initializations"]] == [
        "2022-01-02T00:00:00Z", "2022-04-03T06:00:00Z",
        "2022-07-07T18:00:00Z", "2022-10-07T06:00:00Z"]
    assert result["source_chosen_idx"] == [4, 369, 751, 1117]
    missing = deepcopy(development)
    missing["initializations"] = [row for row in missing["initializations"]
                                  if row["initialization_time"][5:7] not in ("10", "12")]
    with pytest.raises(ValueError, match="every quarter"):
        pilot.pilot_cohort(missing)


def test_model_mapping_keeps_final_m1_and_common_fresh_raw_endpoints(tmp_path, monkeypatch):
    def selected(path):
        family = "gc_mamba" if Path(path).name == "selected_arms_gc_mamba.json" else "full_gc"
        prefix = "gcm" if family == "gc_mamba" else "gcft"
        tasks = []
        for weight in (0., .25, .5, .75, 1.):
            logical = f"{prefix}_s22_l{round(weight * 100):03d}"
            run = logical + ("_lr3em7_fresh" if family == "full_gc" and weight == 0 else
                             "_lr0" if weight == 0 else "_profile" if weight == .5 else "")
            tasks.append({"id": run, "logical_id": logical, "lambda": weight,
                          "learning_rate": 3e-7 if family == "full_gc" and weight == 0 else 3e-5})
        return {"tasks": tasks}

    monkeypatch.setattr(pilot, "unseal", selected)
    monkeypatch.setattr(pilot, "pinned", lambda path: {"path": str(path), "sha256": "fixture"})
    models = pilot.model_records(tmp_path)
    assert len(models) == len({row["id"] for row in models}) == 12
    assert models[0]["id"] == "M1_raw"
    assert models[0]["checkpoint"]["path"].endswith("/checkpoints/checkpoint_step00002000.pkl")
    assert models[1]["id"] == "M1_swa"
    assert models[1]["checkpoint"]["path"].endswith("/swa/swa_step00500-02000.pkl")
    assert all(row["total_training_updates"] == 2500 for row in models[:2])
    for family in ("gc_mamba", "full_gc"):
        fresh = [row for row in models[2:] if row["model_family"] == family]
        assert [row["lambda"] for row in fresh] == [0., .25, .5, .75, 1.]
        assert all(row["step"] == 1250 for row in fresh)
        assert all(row["checkpoint"]["path"].endswith(
            f"/{row['training_run_id']}/checkpoints/checkpoint_step00001250.pkl") for row in fresh)


def spectra_fixture(tmp_path):
    cohort = {"initializations": [{"initialization_time": f"2022-{month:02d}-02T00:00:00Z"}
                                   for month in (1, 4, 7, 10)]}
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(cohort))
    manifest = {"record_hash": "campaign", "source_digest": "source",
                "cohort": {"path": str(path), "sha256": "cohort"}}
    model = {"id": "M1_raw", "checkpoint": {"sha256": "checkpoint"}}
    dest = tmp_path / "evaluation"
    directory = dest / "spectra"
    directory.mkdir(parents=True)
    for month, power in zip((1, 4, 7, 10), (1., 4., 9., 16.)):
        identity = f"2022{month:02d}02T000000Z"
        # Unequal date powers distinguish pooling powers from averaging ratios.
        truth = np.full((40, 2, 180), power)
        prediction = np.stack((truth, np.full_like(truth, 2.)), axis=1)
        meta = {"campaign_hash": "campaign", "source_digest": "source",
                "checkpoint_sha256": "checkpoint", "cohort_sha256": "cohort",
                "model_id": "M1_raw", "initialization_id": identity,
                "initialization_time": f"2022-{month:02d}-02T00:00:00Z",
                "status": "complete", "lead_hours": list(range(6, 241, 6)),
                "branches": ["public_gc", "M1_raw"], "lmax": 179,
                "layout": {"channels": [["2m_temperature", None], ["geopotential", 500]]}}
        np.savez_compressed(directory / f"{identity}.npz", truth_power=truth,
                            prediction_power=prediction, error_power=prediction + 1.,
                            cross_power=-prediction, metadata=json.dumps(meta))
    return dest, manifest, model


def rewrite_spectrum(path, *, metadata=None, array=None):
    with np.load(path, allow_pickle=False) as stored:
        content = {key: stored[key].copy() for key in stored.files}
    if metadata is not None:
        meta = json.loads(str(content["metadata"]))
        meta.update(metadata)
        content["metadata"] = np.asarray(json.dumps(meta))
    if array is not None:
        key, value = array
        content[key] = value
    np.savez_compressed(path, **content)


def test_spectrum_pooling_preserves_power_units_and_negative_cross_power(tmp_path):
    dest, manifest, model = spectra_fixture(tmp_path)
    result = pilot.pool_spectra(dest, manifest, model)
    with np.load(result, allow_pickle=False) as pooled:
        np.testing.assert_allclose(pooled["truth_power"], 7.5)
        np.testing.assert_allclose(pooled["prediction_power"][:, 1], 2.)
        np.testing.assert_allclose(pooled["prediction_power"][:, 1] / pooled["truth_power"], 2. / 7.5)
        np.testing.assert_allclose(pooled["cross_power"][:, 1], -2.)
        metadata = json.loads(str(pooled["metadata"]))
        assert metadata["count"] == 4
        assert metadata["initialization_ids"] == [f"2022{month:02d}02T000000Z" for month in (1, 4, 7, 10)]
        assert metadata["checkpoint_sha256"] == "checkpoint"
        assert metadata["cohort_sha256"] == "cohort"
        assert metadata["source_digest"] == "source"
        assert metadata["branches"] == ["public_gc", "M1_raw"]


@pytest.mark.parametrize("metadata", [
    {"checkpoint_sha256": "wrong-checkpoint"},
    {"campaign_hash": "wrong-campaign"},
    {"cohort_sha256": "wrong-cohort"},
    {"source_digest": "wrong-source"},
    {"lmax": 178},
    {"lead_hours": list(range(6, 235, 6))},
    {"branches": ["M1_raw", "public_gc"]},
    {"status": "partial"},
])
def test_spectrum_pooling_rejects_mismatched_evidence(tmp_path, metadata):
    dest, manifest, model = spectra_fixture(tmp_path)
    rewrite_spectrum(next((dest / "spectra").glob("*.npz")), metadata=metadata)
    with pytest.raises(ValueError, match="provenance/coverage"):
        pilot.pool_spectra(dest, manifest, model)


def test_spectrum_pooling_rejects_missing_dates_and_nonfinite_values(tmp_path):
    dest, manifest, model = spectra_fixture(tmp_path)
    path = next((dest / "spectra").glob("*.npz"))
    original = path.read_bytes()
    path.unlink()
    with pytest.raises(ValueError, match="spectrum dates"):
        pilot.pool_spectra(dest, manifest, model)
    path.write_bytes(original)
    rewrite_spectrum(path, array=("truth_power", np.full((40, 2, 180), np.nan)))
    with pytest.raises(ValueError, match="Nonfinite"):
        pilot.pool_spectra(dest, manifest, model)
