"""Paired AMSE evaluation validates coverage without hiding baseline drift."""
import copy
import json

import numpy as np
import pytest

from scripts.analyze_models import evaluate_v24_amse_pair as workflow


@pytest.mark.parametrize("version", [2, 3])
def test_frozen_metadata_supports_both_zarr_formats(tmp_path, version):
    name = ".zmetadata" if version == 2 else "zarr.json"
    metadata = ({"zarr_consolidated_format": 1, "metadata": {".zgroup": {"zarr_format": 2}}}
                if version == 2 else {"zarr_format": 3, "consolidated_metadata": {
                    "kind": "inline", "metadata": {"temperature": {"node_type": "array"}}}})
    path = tmp_path / name
    path.write_text(json.dumps(metadata))
    record = workflow.frozen_zarr_metadata(tmp_path)
    assert record["path"] == str(path)
    workflow.check_file(record)
    path.write_text("{}")
    with pytest.raises(ValueError, match="Frozen input changed"):
        workflow.check_file(record)


def test_unconsolidated_v3_is_rejected(tmp_path):
    (tmp_path / "zarr.json").write_text(json.dumps({"zarr_format": 3}))
    with pytest.raises(ValueError, match="consolidated Zarr v3"):
        workflow.frozen_zarr_metadata(tmp_path)


def test_preflight_checks_exact_dataset_even_when_spectral_metadata_exists(tmp_path, monkeypatch):
    pilot = {"data_path": str(tmp_path / "spectral")}
    pilot["protocol_hash"] = workflow.protocol_digest(pilot)
    (tmp_path / "pilot.json").write_text(json.dumps(pilot))
    monkeypatch.setattr(workflow, "load_experiment", lambda _: (
        {"spectral_protocol": "pilot.json", "exact_data_path": "missing_exact"}, tmp_path))
    with pytest.raises(FileNotFoundError, match="missing_exact"):
        workflow.validate_evaluation_data(tmp_path)


def result_for(spec, *, baseline=1., ratio=.8):
    reference = np.linspace(baseline, baseline * 2, 40)
    result = {
        "evaluation_status": "complete", "target_steps": 40,
        "evaluated_samples": spec["n_samples"],
        **{key: spec[key] for key in ("protocol_hash", "model_id", "checkpoint_sha256")},
        "baseline_checkpoint_sha256": spec["baseline"]["checkpoint_sha256"],
        "original_graphcast_loss": {
            "baseline_per_step": reference.tolist(), "full_per_step": (reference * ratio).tolist(),
            "baseline_rollout": float(reference.mean()), "full_rollout": float(reference.mean() * ratio),
        },
    }
    if spec["stage"] == "exact":
        result["chosen_idx"] = spec["chosen_idx"]
    else:
        result["initialization_manifest_sha256"] = spec["initialization_manifest_sha256"]
        result["original_graphcast_loss_per_initialization"] = [
            {"initialization_time": stamp, "baseline_per_step": reference.tolist(),
             "full_per_step": (reference * ratio).tolist()}
            for stamp in spec["initialization_times"]
        ]
    return result


def simple_spec(stage="exact"):
    spec = {"stage": stage, "protocol_hash": "spec", "model_id": "amse_step2000",
            "checkpoint_sha256": "ckpt", "baseline": {"checkpoint_sha256": "baseline"},
            "n_samples": 32 if stage == "exact" else 8,
            "chosen_idx": list(range(32)), "initialization_manifest_sha256": "dates"}
    spec["initialization_times"] = [f"2023-{month:02d}-15T00:00:00Z" for month in (1, 2, 4, 5, 7, 8, 10, 11)]
    return spec


@pytest.mark.parametrize("defect", ["partial", "samples", "leads", "anchors", "identity", "aggregate", "nan"])
def test_exact_validation_rejects_incomplete_or_mismatched_evaluation(defect):
    spec = simple_spec()
    result = result_for(spec)
    if defect == "partial":
        result["evaluation_status"] = "partial"
    elif defect == "samples":
        result["evaluated_samples"] = 31
    elif defect == "leads":
        result["original_graphcast_loss"]["full_per_step"].pop()
    elif defect == "anchors":
        result["chosen_idx"] = list(reversed(spec["chosen_idx"]))
    elif defect == "identity":
        result["checkpoint_sha256"] = "wrong"
    elif defect == "aggregate":
        result["original_graphcast_loss"]["full_rollout"] *= 2
    else:
        result["original_graphcast_loss"]["full_per_step"][0] = None
    with pytest.raises((ValueError, AssertionError)):
        workflow.exact_arrays(result, spec)


def test_spectral_dates_and_aggregation_are_checked():
    spec = simple_spec("spectral")
    result = result_for(spec)
    assert workflow.exact_arrays(result, spec)["baseline"].shape == (40,)
    bad = copy.deepcopy(result)
    bad["original_graphcast_loss_per_initialization"].reverse()
    with pytest.raises(ValueError, match="dates differ"):
        workflow.exact_arrays(bad, spec)
    bad = copy.deepcopy(result)
    bad["original_graphcast_loss_per_initialization"][0]["full_per_step"][0] *= 2
    with pytest.raises(AssertionError):
        workflow.exact_arrays(bad, spec)


def test_baseline_drift_is_audited_and_does_not_change_paired_denominator():
    spec = simple_spec()
    values = workflow.exact_arrays(result_for(spec, baseline=1.2), spec)
    drift = workflow.baseline_drift(values["baseline"], np.linspace(1., 2., 40))
    assert drift["outside_tolerance"] == 40
    assert drift["max_relative_difference_pct"] == pytest.approx(20.)
    assert 100 * (1 - values["full"].mean() / values["baseline"].mean()) == pytest.approx(20.)


def test_power_error_and_signed_coherence_use_pooled_statistics():
    # Pooled sufficient statistics: two starts with truth powers 1 and 9,
    # forecast powers .25 and 9, and a quarter-turn / opposite phase example.
    record = {"truth_power": np.array([0., 5., 1.]),
              "baseline_power": np.array([0., 5., 1.]),
              "full_power": np.array([0., 4.625, 1.]),
              "baseline_cross_power": np.array([0., 4., 1.]),
              "full_cross_power": np.array([0., 0., -1.])}
    curves = workflow.spectral_curves(record)
    assert curves["full"]["relative_power_error_pct"][1] == pytest.approx(7.5)
    assert curves["full"]["coherence"][1:] == pytest.approx([0., -1.])
    assert np.isnan(curves["full"]["coherence"][0])
    assert np.isnan(curves["full"]["relative_power_error_pct"][0])


@pytest.fixture
def completed_pair(tmp_path):
    experiment = tmp_path / "pair"
    entries = [{"name": arm, "config": str(experiment / "config.json")} for arm in workflow.ARMS]
    workflow.write_json(experiment / "manifest.json", {"workspace": str(tmp_path), "entries": entries})
    artifact = experiment / "fixture.bin"
    artifact.write_bytes(b"immutable fixture")
    identity = workflow.frozen_file(artifact)
    for stage in ("exact", "spectral"):
        for ai, arm in enumerate(workflow.ARMS):
            for ti, tag in enumerate(workflow.TAGS):
                spec = simple_spec(stage)
                spec.update(arm=arm, tag=tag, model_id=f"{arm}_{tag}", checkpoint=str(artifact),
                            checkpoint_sha256=identity["sha256"], run_config=identity,
                            experiment_manifest=workflow.frozen_file(experiment / "manifest.json"),
                            data_metadata=identity, statistics=[],
                            baseline={"path": str(artifact), "checkpoint_sha256": identity["sha256"]},
                            lead_hours=list(range(6, 241, 6)), spectral_variables=["T850"],
                            spectral_metadata={"fixture": True})
                spec["protocol_hash"] = workflow.protocol_digest(spec)
                directory = experiment / "eval" / stage / arm / tag
                workflow.write_json(directory / "specification.json", spec)
                workflow.write_json(directory / "evaluation.json", result_for(spec, baseline=1 + ai * .2 + ti * .01))
                if stage == "exact":
                    continue
                shape = (40, 1, 4)
                truth = np.ones(shape) * np.array([0., 4., 2., 1.])
                baseline = truth * (1 + ai * .2 + ti * .01)
                full = truth * (.7 + ai * .1)
                arrays = {"truth_power": truth, "baseline_power": baseline, "full_power": full}
                for branch, power in (("baseline", baseline), ("full", full)):
                    cross = np.sqrt(truth * power) * .5
                    arrays[f"{branch}_cross_power"] = cross
                    arrays[f"{branch}_error_power"] = truth + power - 2 * cross
                for si, stamp in enumerate(spec["initialization_times"]):
                    metadata = {**{key: spec[key] for key in ("protocol_hash", "model_id", "checkpoint_sha256")},
                                "baseline_checkpoint_sha256": identity["sha256"], "status": "complete",
                                "spectral_metadata": spec["spectral_metadata"], "initialization_time": stamp}
                    np.savez(directory / f"{si}.npz", **arrays, metadata=json.dumps(metadata),
                             wavenumber=np.arange(4), lead_hours=spec["lead_hours"], variables=["T850"])
    return experiment


def test_report_produces_both_checkpoint_comparisons_and_explicit_drift(completed_pair):
    summary = workflow.report(completed_pair, make_plots=False)
    assert summary["status"] == "complete_with_baseline_drift"
    assert len(summary["exact_graphcast_loss"]) == 8
    for row in summary["exact_graphcast_loss"].values():
        assert row["improvement_pct_rollout"] == pytest.approx(20.)
    assert summary["exact_graphcast_loss"]["exact/amse_step2000"]["baseline_drift_vs_mse_step2000"]["outside_tolerance"] == 40
    assert (completed_pair / "eval/report/README.md").exists()
    assert len(summary["checkpoint_mapping"]) == 8
    data_dir = completed_pair.parent / "plots/analyze_models/data/resolution_eval/pair"
    with np.load(data_dir / "pooled_spectra.npz") as arrays:
        assert "amse_step2000_full_cross_power" in arrays
    assert "amse_step2000" in (data_dir / "spectral_curves.csv").read_text()


def test_report_rejects_missing_spectrum_initialization(completed_pair):
    (completed_pair / "eval/spectral/amse/step2000/0.npz").unlink()
    with pytest.raises(ValueError, match="Incomplete spectral initializations"):
        workflow.report(completed_pair, make_plots=False)


def test_report_rejects_tampered_truth(completed_pair):
    path = completed_pair / "eval/spectral/amse/step2000/0.npz"
    with np.load(path) as source:
        arrays = dict(source)
    arrays["truth_power"] = arrays["truth_power"] * 1.1
    for branch in ("baseline", "full"):
        arrays[f"{branch}_error_power"] = arrays["truth_power"] + arrays[f"{branch}_power"] - 2 * arrays[f"{branch}_cross_power"]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="ERA5 truth differs"):
        workflow.report(completed_pair, make_plots=False)


def test_recovered_report_uses_new_destinations(completed_pair, tmp_path):
    destination = tmp_path / "recovery/report"
    images = tmp_path / "recovery/images"
    summary = workflow.report(completed_pair, make_plots=False,
                              output_dir=destination, output_image_dir=images)
    assert summary["data_dir"] == str(destination)
    assert (destination / "comparison.json").exists()
    assert not (completed_pair / "eval/report").exists()


def test_plot_generation_handles_two_real_arm_labels(completed_pair):
    summary = workflow.report(completed_pair)
    images = completed_pair.parent / "plots/analyze_models/images/resolution_eval/pair"
    for filename in ("power.pdf", "relative_power_error_pct.pdf", "coherence.pdf", "exact_graphcast_loss.pdf",
                     "relative_power_error_pct_step2000_T850.png"):
        assert (images / filename).stat().st_size > 1000
    assert summary["status"] == "complete_with_baseline_drift"


def test_evaluation_workers_run_in_fresh_processes(tmp_path, monkeypatch):
    workflow.write_json(tmp_path / "manifest.json", {"entries": [{"name": arm} for arm in workflow.ARMS]})
    commands = []
    monkeypatch.setattr(workflow.subprocess, "run", lambda argv, **kwargs: commands.append(argv))
    workflow.evaluate(tmp_path, 1)
    assert len(commands) == 5
    assert commands[0][2] == "swa"
    assert [command[command.index("--stage") + 1] for command in commands[1:]] == ["exact", "exact", "spectral", "spectral"]
    assert [command[command.index("--tag") + 1] for command in commands[1:]] == list(workflow.TAGS) * 2


def test_smoke_cli_requires_explicit_checkpoint(tmp_path):
    with pytest.raises(SystemExit):
        workflow.main(["smoke", "--experiment", str(tmp_path), "--task", "0"])
