"""Diagnostics must preserve native updates and the initial reference on resume."""
from collections import namedtuple
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.experiments import train_v24_res1_fresh_long as driver


def test_parameter_groups_and_zero_reference():
    assert driver.parameter_group("mesh_gnn/processor_nodes_1_mesh_nodes_mlp/~/linear_0") == "mesh_processor_1"
    assert driver.parameter_group("mesh_gnn/encoder_edges_mesh_layer_norm") == "mesh_edge_embedding"
    assert driver.parameter_group("mesh_interleaved_temporal_r0_s1/~_run_sequence/mamba_block_0") == "mesh_interleaved_temporal_r0_s1"
    initial = {"temporal_residual_head": {"w": np.zeros((1, 2))}}
    after = {"temporal_residual_head": {"w": np.array([[3., 4.]])}}
    result = driver.parameter_metrics(initial, initial, after)["groups"]["correction_head"]["all"]
    assert result["update_norm"] == 5
    assert result["displacement_over_initial"] is None
    assert result["update_over_before"] is None


def test_adam_gradient_estimate_and_ambiguous_moments():
    Adam = namedtuple("Adam", "mu nu")
    before = {"grid2mesh_gnn/test": {"w": np.array([1., 2.])}}
    after = {"grid2mesh_gnn/test": {"w": np.array([1.2, 2.1])}}
    mu_before = {"grid2mesh_gnn/test": {"w": np.array([.1, .2])}}
    mu_after = {"grid2mesh_gnn/test": {"w": np.array([.39, .58])}}
    assert driver.adam_moment((Adam(mu_before, {}), Adam(mu_after, {}))) is None
    np.testing.assert_array_equal(driver.adam_moment(((), Adam(mu_before, {})))["grid2mesh_gnn/test"]["w"], [.1, .2])
    result = driver.parameter_metrics(before, before, after, mu_before=mu_before,
                                      mu_after=mu_after, beta1=.9)
    assert result["groups"]["grid_to_mesh"]["all"]["post_clip_gradient_norm_estimate"] == pytest.approx(5)


def test_wrapped_updates_unchanged_and_resume_uses_original_reference(tmp_path):
    config = SimpleNamespace(run_dir=tmp_path, checkpoint_every=1000, max_steps=11, adam_beta1=.9)
    diagnostics = driver.TrainingDiagnostics(config)
    params = {"temporal_residual_head": {"w": np.zeros((1, 1), dtype=np.float32)}}
    returned = []

    def step(parameters, state, optimizer_state, *args):
        result = ({m: {n: a + 1 for n, a in v.items()} for m, v in parameters.items()},
                  state, optimizer_state, 1., 2., np.array([1.]))
        returned.append(result)
        return result

    wrapped = diagnostics.wrap(step)
    for _ in range(10):
        result = wrapped(params, {}, ())
        assert result is returned[-1]
        params = result[0]
    assert diagnostics.snapshot_path(0).is_file()
    assert diagnostics.snapshot_path(1).is_file()
    assert diagnostics.snapshot_path(10).is_file()
    assert not diagnostics.snapshot_path(2).exists()
    resumed = driver.TrainingDiagnostics(config, completed_step=10)
    resumed.wrap(step)(params, {}, ())
    records = [json.loads(line) for line in (tmp_path / "diagnostics/parameter_metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == [0, 1, 10, 11]
    last = records[-1]["groups"]["correction_head"]["all"]
    assert last["displacement_norm"] == 11
    assert last["update_norm"] == 1
    assert last["displacement_over_initial"] is None


def test_runner_hook_restored_after_failure():
    original = lambda **kwargs: None
    runner = SimpleNamespace(make_train_step=original)
    with pytest.raises(RuntimeError):
        with driver.instrument_runner(runner, SimpleNamespace(wrap=lambda x: x)):
            assert runner.make_train_step is not original
            raise RuntimeError("training failure")
    assert runner.make_train_step is original


def test_resume_archives_unsaved_diagnostics_without_losing_initial(tmp_path):
    config = SimpleNamespace(run_dir=tmp_path, checkpoint_every=1000, max_steps=20, adam_beta1=.9)
    diagnostics = driver.TrainingDiagnostics(config)
    params = {"temporal_residual_head": {"w": np.zeros((1, 1))}}
    for step in (0, 1, 10):
        diagnostics.snapshot(step, params)
    metrics = diagnostics.directory / "parameter_metrics.jsonl"
    original = "".join(json.dumps(dict(step=step)) + "\n" for step in (0, 1, 10)) + '{"step":11'
    metrics.write_text(original)
    resumed = driver.TrainingDiagnostics(config, completed_step=1)
    resumed.wrap(lambda *args: args)
    assert [json.loads(line)["step"] for line in metrics.read_text().splitlines()] == [0, 1]
    assert resumed.snapshot_path(0).is_file()
    assert resumed.snapshot_path(1).is_file()
    assert not resumed.snapshot_path(10).exists()
    archives = list((diagnostics.directory / "resume_archives").iterdir())
    assert len(archives) == 1
    assert (archives[0] / "parameter_metrics.jsonl").read_text() == original
    assert (archives[0] / resumed.snapshot_path(10).name).is_file()
    resumed.recover_resume_history()
    assert len(list((diagnostics.directory / "resume_archives").iterdir())) == 1
