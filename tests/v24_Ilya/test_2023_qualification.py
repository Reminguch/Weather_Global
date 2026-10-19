"""Qualification cannot pass from a flag, absent evidence, or changed evidence."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.v24_2023_qualification import (
    REQUIRED_CHECKS, build_numerical_qualification, compare_losses, file_record,
    validate_numerical_qualification, development_initialization_times,
)


MODELS = ["M1", "M2", "M3", "M1_reset", "C_carry", "C_reset", "R_reset"]
TIMES = [f"2022-{month:02d}-15T00:00:00Z" for month in (1, 4, 7, 10)]


def protocol(tmp_path):
    data_path = tmp_path / "historical.zarr"
    times = np.arange(np.datetime64("2022-01-01"), np.datetime64("2023-01-01"), np.timedelta64(6, "h"))
    (data_path / "time").mkdir(parents=True)
    metadata = {"time/.zarray": {"shape": [len(times)], "chunks": [len(times)], "dtype": "<i8",
                                "compressor": None, "filters": None, "order": "C", "zarr_format": 2},
                "time/.zattrs": {"units": "hours since 2022-01-01", "calendar": "proleptic_gregorian",
                                  "_ARRAY_DIMENSIONS": ["time"]}}
    (data_path / ".zmetadata").write_text(json.dumps({"metadata": metadata, "zarr_consolidated_format": 1}))
    (data_path / "time/0").write_bytes((np.arange(len(times), dtype="<i8") * 6).tobytes())
    return {"protocol_hash": "frozen", "project_root": str(tmp_path),
            "models": [{"model_id": m, "checkpoint_sha256": m, "checkpoint": str(tmp_path/m)} for m in MODELS],
            "pilot": {"initialization_times": TIMES, "data_path": str(data_path)}, "baseline": {"checkpoint_sha256": "baseline"},
            "numerical_qualification": {"path": str(tmp_path/"numerical_qualification.json")}}


def fingerprint(empty=False):
    return {"sha256": "a" * 64, "leaves": [] if empty else [{"name": "field", "shape": [1], "dtype": "float32"}]}


def probe(model="M1", mode="interleaved", pid=1, gpu="gpu-a", times=TIMES, repeat=0):
    observation = {"setup": {k: fingerprint(k.endswith("state") or (k == "residual_params" and mode == "standalone")) for k in
                            ("baseline_params", "baseline_state", "residual_params", "residual_state")},
                   "initializations": [{"initialization_time": t, "inputs": fingerprint(), "rng": fingerprint()} for t in times],
                   "predictions": [{"initialization_time": t, "step_index": lead, **{k: "a" * 64 for k in ("truth", "baseline", "full")}}
                                   for t in times for lead in range(40)],
                   "forcings": [{"initialization_time": t, "step_index": lead, **fingerprint()} for t in times for lead in range(40)]}
    observation["setup"].update(
        statistics={k: fingerprint() for k in ("mean_by_level", "stddev_by_level", "diffs_stddev_by_level")},
        baseline_model_config={"resolution": 1}, task_config={"input_duration": "12h"})
    return {"protocol_hash": "frozen", "model_id": model, "checkpoint_sha256": model,
            "baseline_checkpoint_sha256": "baseline", "diagnostic_mode": mode,
            "process_run_id": f"process-{pid}-{gpu}", "repeat_index": repeat, "rs_reset_every_step": False,
            "pid": pid, "hostname": "host", "gpu_uuid": gpu, "gpu_kind": "A100", "driver_version": "driver",
            "numerical_observation": observation, "evaluation_status": "complete", "metrics_complete": True,
            "evaluated_samples": len(times), "target_steps": 40,
            "original_graphcast_loss_per_initialization": [
                {"initialization_time": t, "baseline_per_step": [2.]*40, "full_per_step": [1.5]*40} for t in times],
            "original_graphcast_loss": {"improvement_pct_rollout": 25., "baseline_per_step": [2.] * 40, "full_per_step": [1.5] * 40}}


def save(path, value):
    path.write_text(json.dumps(value))
    return path


def test_pending_and_forged_checks_cannot_release(tmp_path):
    p = protocol(tmp_path)
    report = build_numerical_qualification(p, [])
    assert report["status"] == "pending"
    assert report["runtime"] is None
    save(tmp_path/"numerical_qualification.json", report)
    with pytest.raises(ValueError, match="pending"):
        validate_numerical_qualification(tmp_path, p)
    fake = copy.deepcopy(report)
    fake["status"] = "passed"
    evidence = file_record(save(tmp_path/"unrelated.json", {}))
    fake["checks"] = {k: {"passed": True, "artifacts": [evidence]} for k in REQUIRED_CHECKS}
    save(tmp_path/"numerical_qualification.json", fake)
    with pytest.raises(ValueError, match="reproduce"):
        validate_numerical_qualification(tmp_path, p)


def all_evidence(tmp_path, p):
    paths = []
    for model in MODELS:
        for repeat in (0, 1):
            paths.append(save(tmp_path/f"{model}_{repeat}.json", probe(model, repeat=repeat)))
    for label, extra in (("fresh", probe(pid=2)), ("gpu", probe(pid=3, gpu="gpu-b")),
                         ("alone", probe(mode="standalone"))):
        paths.append(save(tmp_path/f"{label}.json", extra))
    indices = list(range(1, 33))
    source = save(tmp_path/"development.json", {"evaluated_samples": 32, "target_steps": 40,
                   "ckpt": p["models"][0]["checkpoint"], "global_chosen_idx": indices, "chosen_idx": indices,
                   "original_graphcast_loss": {"improvement_pct_rollout": 24., "baseline_per_step": [2.] * 40, "full_per_step": [1.52] * 40}})
    times = development_initialization_times(p, json.loads(source.read_text()))
    bridge = probe(mode="bridge", times=times)
    bridge.update(bridge_source=file_record(source), bridge_source_chosen_indices=indices)
    paths.append(save(tmp_path/"bridge.json", bridge))
    return paths


def test_complete_qualification_and_changed_artifact(tmp_path):
    p = protocol(tmp_path)
    paths = all_evidence(tmp_path, p)
    report = build_numerical_qualification(p, paths)
    assert report["status"] == "passed"
    assert report["runtime"] == {"gpu_kind": "A100", "driver_version": "driver"}
    assert set(report["checks"]) == set(REQUIRED_CHECKS)
    save(tmp_path/"numerical_qualification.json", report)
    assert validate_numerical_qualification(tmp_path, p)["status"] == "passed"
    altered = json.loads(paths[0].read_text())
    altered["original_graphcast_loss_per_initialization"][0]["full_per_step"][39] = 1.8
    save(paths[0], altered)
    with pytest.raises(ValueError, match="changed"):
        validate_numerical_qualification(tmp_path, p)
    assert build_numerical_qualification(p, paths)["status"] == "failed"


def test_no_test_year_numerical_probes_and_strict_tolerance(tmp_path):
    p = protocol(tmp_path)
    bad = save(tmp_path/"bad.json", probe(times=["2023-01-01T00:00:00Z"]))
    with pytest.raises(ValueError, match="only 2022"):
        build_numerical_qualification(p, [bad])
    a, b = probe(), probe()
    b["original_graphcast_loss_per_initialization"][0]["baseline_per_step"][0] *= 1.002
    result = compare_losses(a, b)
    assert not result["passed"]
    assert result["branches"]["baseline"]["first_differing_index"] == [[0, 0]]


@pytest.mark.parametrize("mutation,match", [
    ("missing_truth", "predictions evidence"), ("missing_forcings", "forcings evidence"),
    ("missing_full_digest", "full prediction"), ("missing_residual", "residual_params"),
    ("changed_residual", "residual parameters/state"), ("mixed_driver", "same GPU kind and driver"),
    ("duplicate_repeat", None), ("bad_reset", "memory policy"),
])
def test_incomplete_or_inconsistent_probe_evidence_cannot_qualify(tmp_path, mutation, match):
    p = protocol(tmp_path)
    paths = all_evidence(tmp_path, p)
    value = json.loads(paths[1].read_text())
    observation = value["numerical_observation"]
    if mutation == "missing_truth":
        observation["predictions"] = []
    elif mutation == "missing_forcings":
        observation.pop("forcings")
    elif mutation == "missing_full_digest":
        observation["predictions"][0].pop("full")
    elif mutation == "missing_residual":
        observation["setup"].pop("residual_params")
    elif mutation == "changed_residual":
        observation["setup"]["residual_state"]["sha256"] = "b" * 64
    elif mutation == "mixed_driver":
        value["driver_version"] = "another-driver"
    elif mutation == "bad_reset":
        value["rs_reset_every_step"] = True
    else:
        value["repeat_index"] = 0
    save(paths[1], value)
    if match is None:
        result = build_numerical_qualification(p, paths)
        assert not result["checks"]["within_process_repeatability"]["passed"]
        assert result["status"] == "failed"
    else:
        with pytest.raises(ValueError, match=match):
            build_numerical_qualification(p, paths)


def test_bridge_dates_are_derived_from_historical_coordinate_not_claimed_indices(tmp_path):
    p = protocol(tmp_path)
    paths = all_evidence(tmp_path, p)
    value = json.loads(paths[-1].read_text())
    original = [r["initialization_time"] for r in value["original_graphcast_loss_per_initialization"]]
    shifted = [np.datetime_as_string(np.datetime64(t.removesuffix("Z")) + np.timedelta64(6, "h"), unit="s") + "Z" for t in original]
    replacement = probe(mode="bridge", times=shifted)
    replacement.update({k: value[k] for k in ("bridge_source", "bridge_source_chosen_indices")})
    save(paths[-1], replacement)
    with pytest.raises(ValueError, match="UTC dates differ"):
        build_numerical_qualification(p, paths)


def test_bridge_effect_is_computed_from_valid_aggregates(tmp_path):
    p = protocol(tmp_path)
    paths = all_evidence(tmp_path, p)
    value = json.loads(paths[-1].read_text())
    value["original_graphcast_loss"]["improvement_pct_rollout"] = 99
    save(paths[-1], value)
    with pytest.raises(ValueError, match="reported improvement"):
        build_numerical_qualification(p, paths)


@pytest.mark.parametrize("branch,value", [("baseline", 0), ("baseline", -1), ("full", -1)])
def test_numerical_loss_comparison_rejects_nonphysical_losses(branch, value):
    a = probe()
    a["original_graphcast_loss_per_initialization"][0][f"{branch}_per_step"][0] = value
    with pytest.raises(ValueError, match="Invalid numerical comparison"):
        compare_losses(a, a)
