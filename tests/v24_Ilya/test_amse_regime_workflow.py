"""Joint selection guards, paired bootstrap, and generic checkpoint protocol."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_models import evaluate_v24_amse_regime as workflow


def simple_spec(stage="exact", n_samples=32):
    return {"stage": stage, "target_steps": 40, "n_samples": n_samples,
            "model_id": "test", "protocol_hash": "protocol", "checkpoint_sha256": "checkpoint",
            "baseline": {"checkpoint_sha256": "gc"}, "chosen_idx": list(range(32)),
            "evaluation_year": 2022 if stage == "exact" else 2023,
            "initialization_manifest_sha256": "dates",
            "initialization_times": [f"2023-01-{i+1:02d}T00:00:00Z" for i in range(n_samples)]}


def simple_result(spec, loss=1.):
    result = {"evaluation_status": "complete", "metrics_complete": True, "remaining_samples": 0,
              "evaluated_samples": spec["n_samples"], "target_steps": 40,
              **{key: spec[key] for key in ("model_id", "protocol_hash", "checkpoint_sha256")},
              "baseline_checkpoint_sha256": spec["baseline"]["checkpoint_sha256"],
              "chosen_idx": spec["chosen_idx"], "initialization_manifest_sha256": "dates",
              "original_graphcast_loss": {"baseline_per_step": [2.]*40, "full_per_step": [loss]*40,
                                          "baseline_rollout": 2., "full_rollout": loss},
              "original_graphcast_loss_per_initialization": [
                  {"initialization_time": stamp, "baseline_per_step": [2.]*40, "full_per_step": [loss]*40}
                  for stamp in spec["initialization_times"]],
              "per_channel_per_step": {channel: {"rmse_baseline": [2.]*40, "rmse_full": [1.]*40}
                                       for channel in workflow.CHANNELS.values()}}
    return result


def test_initialization_selection_reads_only_exact32(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "verify_specification", lambda spec: None)
    experiments = [tmp_path/"old", tmp_path/"new"]
    for ei, experiment in enumerate(experiments):
        for ti, tag in enumerate(workflow.TAGS):
            directory = experiment/"eval/exact/mse_fp32"/tag
            directory.mkdir(parents=True)
            ckpt = experiment/f"{tag}.pkl"
            ckpt.write_bytes(f"weights {ei} {ti}".encode())
            config = experiment/"run_config.json"
            workflow.write_json(config, {"architecture": {}, "objective": {"loss_kind": "mse_fp32"}})
            spec = simple_spec()
            spec.update(checkpoint=str(ckpt), checkpoint_sha256=workflow.frozen_file(ckpt)["sha256"],
                        run_config=workflow.frozen_file(config), architecture={}, statistics=[], data_path="data",
                        data_metadata={}, forecast={}, lead_hours=list(range(6, 241, 6)))
            workflow.write_json(directory/"specification.json", spec)
            workflow.write_json(directory/"evaluation.json", simple_result(spec, 1.2-ei*.1-ti*.03))
            # Any accidental spectral access would encounter invalid JSON.
            spectral = experiment/"eval/spectral/mse_fp32"/tag
            spectral.mkdir(parents=True)
            (spectral/"evaluation.json").write_text("DO NOT READ 2023")
    selection = workflow.select_initialization(experiments)
    assert selection["selected"]["experiment"] == str(experiments[1])
    assert selection["selected"]["tag"] == workflow.TAGS[1]
    assert len(selection["candidates"]) == 4


@pytest.mark.parametrize("defect", ["partial", "dates", "missing_channel", "wrong_checkpoint"])
def test_generic_spectral_validation_rejects_bad_inputs(defect):
    spec = simple_spec("spectral", 12)
    result = simple_result(spec)
    if defect == "partial":
        result["remaining_samples"] = 1
    elif defect == "dates":
        result["original_graphcast_loss_per_initialization"].reverse()
    elif defect == "missing_channel":
        result["per_channel_per_step"].pop("temperature_level850")
    else:
        result["checkpoint_sha256"] = "another"
    with pytest.raises((ValueError, KeyError)):
        workflow.validate_result(result, spec)


def test_generic_spectral_validation_accepts_more_than_eight_starts():
    spec = simple_spec("spectral", 12)
    assert workflow.validate_result(simple_result(spec), spec)["full"].shape == (40,)


def test_reuse_rejects_changed_origins_even_same_checkpoint():
    spec = simple_spec("spectral", 8)
    spec.update(architecture={}, statistics=[], forecast={}, data_path="data", data_metadata={},
                lead_hours=list(range(6, 241, 6)), warmup_steps=0, run_config={"sha256": "cfg"},
                spectral_variables=["T850"], spectral_metadata={})
    workflow.compatible_reuse(spec, copy.deepcopy(spec))
    other = copy.deepcopy(spec)
    other["initialization_times"][0] = "2023-02-01T00:00:00Z"
    with pytest.raises(ValueError, match="initialization_times"):
        workflow.compatible_reuse(spec, other)


def rows(rmse=1., power=60., coherence=.7):
    return [{"feature": feature, "day": day, "rmse": rmse,
             "mean_relative_power_error_pct": power, "mean_coherence": coherence}
            for feature in workflow.FEATURES for day in workflow.DAYS]


def test_aggregate_win_does_not_hide_single_field_rmse_harm():
    candidate = rows(.9, 50.)
    next(row for row in candidate if row["feature"] == "T850" and row["day"] == 10)["rmse"] = 1.03
    score = workflow.score_candidate(candidate, rows(), rows(.8, 70.), .9)
    assert score["aggregate_qualified"]
    assert score["joint_wins_vs_gc"] == 35
    assert not score["eligible"]
    assert score["safety_guard_failures"][0]["feature"] == "T850"


def test_no_qualifier_requests_feedback_and_does_not_name_winner():
    bad = {"id": "bad", "score": workflow.score_candidate(rows(1.1, 40.), rows(), rows(.8, 70.), 1.1)}
    finalists, feedback = workflow.rank_finalists([bad])
    assert finalists == [] and feedback


def test_ranking_retains_distinct_run_families_and_coherence_guard():
    candidates = []
    for name, directory, rmse, power, coherence in (("a100", "a", .9, 50., .7),
                                                  ("a200", "a", .89, 49., .7),
                                                  ("b100", "b", .91, 45., .69)):
        candidates.append({"id": name, "source_run_dir": directory,
                           "score": workflow.score_candidate(rows(rmse, power, coherence), rows(), rows(.8, 70.), .9)})
    finalists, feedback = workflow.rank_finalists(candidates)
    assert {row["source_run_dir"] for row in finalists} == {"a", "b"}
    assert feedback


def test_paired_block_weights_preserve_blocks_and_season_weights():
    grouping = [{"group": "a", "stratum": "winter"}, {"group": "a", "stratum": "winter"},
                {"group": "b", "stratum": "winter"}, {"group": "c", "stratum": "summer"}]
    weights = workflow.grouped_weights(grouping, 50, 4)
    np.testing.assert_array_equal(weights[:, 0], weights[:, 1])
    np.testing.assert_allclose(weights[:, :3].sum(axis=1), .75)
    np.testing.assert_allclose(weights[:, 3], .25)
    np.testing.assert_array_equal(weights, workflow.grouped_weights(grouping, 50, 4))


def test_bootstrap_refuses_global_rmse_reconstruction():
    result = workflow.bootstrap_joint({"global_mse": None}, {"global_mse": None}, "baseline", [])
    assert result["status"] == "unavailable"


def bootstrap_record(power, mse, exact):
    n = 4
    shape = (n, 40, 9, 4)
    truth = np.ones(shape)
    prediction = truth*power
    truth[..., 0] = prediction[..., 0] = 0
    return {"global_mse": {branch: np.full((n, 40, 9), mse) for branch in ("baseline", "full")},
            "times": [str(i) for i in range(n)], "variables": list(workflow.FEATURES),
            "stacked": {"truth_power": truth, "full_power": prediction, "baseline_power": prediction,
                        "full_cross_power": np.sqrt(truth*prediction)*.8,
                        "baseline_cross_power": np.sqrt(truth*prediction)*.8},
            "result": {"original_graphcast_loss_per_initialization": [
                {"full_per_step": [exact]*40, "baseline_per_step": [exact]*40} for _ in range(n)]}}


def test_paired_bootstrap_recomputes_root_and_pooled_power_ratios(monkeypatch):
    monkeypatch.setattr(workflow, "_time", lambda value: value)
    candidate = bootstrap_record(.81, .64, .6)
    reference = bootstrap_record(.36, 1., 1.)
    grouping = [{"initialization_time": str(i), "group": str(i//2), "stratum": "all"} for i in range(4)]
    result = workflow.bootstrap_joint(candidate, reference, "baseline", grouping,
                                      {**workflow.POLICY, "bootstrap_replicates": 20})
    np.testing.assert_allclose(result["percentile95"]["mean_rmse_ratio"], [.8, .8])
    np.testing.assert_allclose(result["percentile95"]["mean_power_error_ratio"], [.19/.64, .19/.64])
    assert result["joint_better_bootstrap_fraction"] == 1.


def test_evaluate_runs_each_stage_in_a_fresh_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "resolve_result", lambda *args: (_ for _ in ()).throw(FileNotFoundError()))
    commands = []
    monkeypatch.setattr(workflow.subprocess, "run", lambda argv, **kwargs: commands.append(argv))
    workflow.evaluate(tmp_path/"manifest.json", 3)
    assert len(commands) == 2
    assert [command[command.index("--stage")+1] for command in commands] == ["exact", "spectral"]
    assert all(command[2] == "worker" for command in commands)


def test_confirmation_report_accepts_spectral_only_and_never_selects(tmp_path, monkeypatch):
    manifest = {"output_dir": str(tmp_path), "canonical_mse_id": "mse", "evaluation_role": "confirmation",
                "task_specs": [{"id": "mse", "label": "MSE", "role": "mse_control"},
                               {"id": "candidate", "label": "AMSE", "role": "candidate"}],
                "protocols": {"spectral": {"initialization_times": ["time"]}},
                "selection_policy": workflow.POLICY, "bootstrap_grouping": [], "scope": "confirmation",
                "observer_factory": None}
    monkeypatch.setattr(workflow, "load_manifest", lambda path: manifest)
    calls = []
    def resolve(path, index, stage):
        calls.append(stage)
        return tmp_path/str(index), {}, {"original_graphcast_loss": {"baseline_rollout": 1., "full_rollout": .9}}, True
    monkeypatch.setattr(workflow, "resolve_result", resolve)
    monkeypatch.setattr(workflow, "load_spectral_result", lambda directory, spec, result: {
        "result": result, "stacked": {"truth_power": np.ones((1, 1, 1, 2))},
        "sources": [], "global_mse": None})
    monkeypatch.setattr(workflow, "feature_rows", lambda record, branch="full": rows())
    monkeypatch.setattr(workflow, "frozen_file", lambda path: {"path": str(path), "sha256": "test"})
    monkeypatch.setattr(workflow, "rank_finalists", lambda *args: (_ for _ in ()).throw(AssertionError("No holdout selection")))
    report = workflow.report(tmp_path/"manifest.json")
    assert calls == ["spectral", "spectral"]
    assert report["finalists"] == [] and report["feedback_needed"] is False
    assert all(row["exact32_loss"] is None for row in report["records"])


def test_global_observer_retains_per_initialization_mse_for_bootstrap(tmp_path, monkeypatch):
    import types
    from src.models.mamba.v24_Ilya import spectral_observer
    monkeypatch.setattr(spectral_observer, "extract_fields", lambda dataset: dataset.values)
    identity = {"model_id": "model", "protocol_hash": "protocol", "checkpoint_sha256": "ckpt"}
    observer = workflow.GlobalFieldObserver(tmp_path, identity, target_steps=1)
    observer.begin_sample({"initialization_id": "sample", "initialization_time": "2023-01-01T00:00:00Z"})
    truth = types.SimpleNamespace(values=np.zeros((9, 3, 2)), lat=np.array([-60., 0., 60.]))
    baseline = types.SimpleNamespace(values=np.ones((9, 3, 2))*2)
    full = types.SimpleNamespace(values=np.ones((9, 3, 2)))
    observer.update_step(0, truth, baseline, full)
    observer.finish_sample()
    result = json.loads((tmp_path/"global_metrics/sample.json").read_text())
    np.testing.assert_allclose(result["baseline_mse"], 4)
    np.testing.assert_allclose(result["full_mse"], 1)
    assert result["features"] == list(workflow.FEATURES)


def test_preparation_supports_future_checkpoints_and_resolvable_64_confirmation_dates(tmp_path, monkeypatch):
    import pandas as pd
    import xarray as xr
    from src.models.mamba.v24_Ilya.data import EvalDataset, resolve_initialization_manifest
    monkeypatch.setattr(workflow, "verify_frozen_zarr_metadata", lambda *args: None)
    reference = tmp_path/"reference"
    for stage in ("exact", "spectral"):
        spec = simple_spec(stage, 32 if stage == "exact" else 8)
        spec.update(data_path="data", data_metadata={}, run_config={}, experiment_manifest={},
                    checkpoint="old", architecture={}, statistics=[], stats_dir="stats", forecast={},
                    warmup_steps=24 if stage == "exact" else 0, lead_hours=list(range(6, 241, 6)))
        workflow.write_json(reference/"eval"/stage/"mse_fp32"/workflow.TAGS[1]/"specification.json", spec)
    starts = [stamp.isoformat().replace("+00:00", "Z") for stamp in pd.date_range("2023-02-01", periods=64, tz="UTC")]
    dates = tmp_path/"dates.json"
    workflow.write_json(dates, {"initialization_times": starts})
    task = {"id": "future", "source_run_dir": str(tmp_path/"future_run"), "checkpoint": str(tmp_path/"future.pkl")}
    output = tmp_path/"confirmation"
    manifest = workflow.prepare_evaluation([task], output, "future", reference, dates, role="confirmation")
    assert "checkpoint_record" not in manifest["task_specs"][0]
    assert manifest["protocols"]["spectral"]["n_samples"] == 64
    dataset = EvalDataset(xr.Dataset(coords={"time": pd.date_range("2022-12-30", "2024-01-11", freq="6h")}),
                          pd.Timedelta("6h"), 2)
    resolved = resolve_initialization_manifest(dataset, output/"initializations.json", target_steps=40)
    assert [row["initialization_time"] for row in resolved] == starts
    assert len(manifest["bootstrap_grouping"]) == 64
    workflow.load_manifest(output/"manifest.json")
    workflow.write_json(dates, {"initialization_times": ["2023-01-01T00:00:00Z"]})
    with pytest.raises(ValueError, match="overlap"):
        workflow.prepare_evaluation([task], tmp_path/"overlap", "future", reference, dates, role="confirmation")


@pytest.mark.parametrize("feature", ["MSLP", "TP6", "Q700"])
def test_all_primitive_features_have_two_percent_rmse_guard(feature):
    candidate = rows(.9, 50.)
    next(row for row in candidate if row["feature"] == feature)["rmse"] = 1.021
    score = workflow.score_candidate(candidate, rows(), rows(.8, 70.), .9)
    assert not score["safety_guard_passed"] and not score["eligible"]


def test_recipe_summary_averages_seed_metrics_and_never_forecast_fields():
    records = []
    for seed, rmse in ((22, .8), (23, .9), (24, 1.)):
        records.append({"id": f"seed{seed}", "seed": seed, "recipe_id": "recipe",
                        "score": workflow.score_candidate(rows(rmse, 50.), rows(), rows(.8, 70.), .9)})
    summary = workflow.aggregate_recipe(records, {}, workflow.POLICY)
    assert summary["n_seeds"] == 3
    assert summary["score"]["per_feature_lead"][0]["rmse"] == pytest.approx(.9)
    assert summary["bootstrap_vs_gc"]["status"] == "unavailable"
    assert "not an ensemble" in summary["aggregation"]


@pytest.mark.parametrize("ramp,weighted", [(0, False), (0, True), (100, False)])
def test_planned_configuration_uses_normalized_omitted_default_contract(tmp_path, ramp, weighted):
    from src.models.mamba.v24_Ilya.training.config import V24IlyaTrainConfig, load_training_config
    from src.models.mamba.v24_Ilya.config import V24IlyaArchitectureConfig
    config = V24IlyaTrainConfig(prepared_root=Path("prepared"), anchor_manifest_root=Path("anchors"),
                               baseline_checkpoint=Path("baseline.npz"), output_root=Path("output"), run_name="test",
                               architecture=V24IlyaArchitectureConfig(),
                               segment_steps=24, bptt_steps=24, ar_tail_k=20, loss_kind="amse", amse_lmax=2,
                               amse_lambda=.5, amse_lambda_ramp_steps=ramp)
    planned = config.to_dict()
    # The launcher emits disabled fields explicitly; checkpoint serialization omits them.
    planned["objective"].update(amse_lambda_start=0., amse_lambda_ramp_steps=ramp)
    if weighted:
        planned["objective"].update(amse_variable_lambda_multipliers={"temperature": .5},
                                     amse_degree_lambda_multipliers=[0., .5, 1.])
    path = tmp_path/"planned.json"
    workflow.write_json(path, planned)
    actual = load_training_config(path).to_dict()
    if not ramp:
        assert "amse_lambda_ramp_steps" not in actual["objective"]
    workflow.validate_planned_config(actual, path)
    actual["objective"]["amse_lambda"] = .25
    with pytest.raises(ValueError, match="amse_lambda"):
        workflow.validate_planned_config(actual, path)
