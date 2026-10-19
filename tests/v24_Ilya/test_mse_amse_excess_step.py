"""Closed-loop replay acceptance commits only accepted candidate state."""
from dataclasses import replace
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from src.models.mamba.v24_Ilya.training.mgda_optimizer import RMSOptimizer
from src.models.mamba.v24_Ilya.training.mgda_step import (
    JointDescentFailure, make_multiobjective_train_step,
)

from test_mgda_step import assert_tree_close, setup


def excess_setup(mode="mgda", learning_rate=1e-4, **policy):
    cfg, transforms, _, params, state, args, _ = setup()
    cfg.objective_pair = "mse_amse_excess"
    cfg.mgda_direction = mode
    cfg.mgda_backtracking = {
        "repeatability_repeats": 3, "absolute_floor": 1e-10, "relative_floor": 0.,
        "max_relative_noise": 1e-3, "max_halvings": 8, "shrink": .5, **policy,
    }
    # Positive proportional toy objectives isolate the acceptance machinery
    # from the spherical-harmonic implementation tested in the loss tests.
    def pair(prediction, truth):
        from test_mgda_step import gc_loss
        loss, _ = gc_loss(prediction, truth)
        return jnp.stack((loss, 2 * loss)), {}
    transforms = replace(transforms, full_field_objectives=pair)
    optimizer = RMSOptimizer(lambda count: jnp.asarray(learning_rate), clip=0.,
                             spatial_multiplier=1., mamba_multiplier=1., direction_mode=mode)
    step = make_multiobjective_train_step(transforms=transforms, optimizer=optimizer,
        baseline_params={}, baseline_state={}, config=cfg, time_step=pd.Timedelta("6h"), input_steps=2)
    return cfg, transforms, optimizer, params, state, args, step


def snapshot(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), tree)


def assert_tree_identical(actual, original):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(original)
    for left, right in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(original), strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("mode", ["mgda", "mse_control", "fixed_sum"])
def test_accepted_step_commits_candidate_replayed_memory_once(mode):
    cfg, _, optimizer, params, state, args, step = excess_setup(mode)
    optimizer_state = optimizer.init(params)
    before = snapshot((params, state, optimizer_state, args[0]))
    original_losses, original_end_memory, _, _ = step.forward_only(params, state, *args)
    result = step(params, state, optimizer_state, *args)
    accepted_losses, accepted_memory, _, _ = step.forward_only(result.params, state, *args)
    assert_tree_close(result.state, accepted_memory)
    assert not np.array_equal(np.asarray(result.state["s"]), np.asarray(original_end_memory["s"]))
    assert int(result.optimizer_state.count) == 1
    assert float(accepted_losses[0]) < float(original_losses[0])
    assert float(accepted_losses[1]) < float(original_losses[1])
    np.testing.assert_allclose([result.diagnostics["objective_1_after"], result.diagnostics["objective_2_after"]],
                               accepted_losses)
    assert result.diagnostics["backtracking_accepted"]
    assert result.diagnostics["backtracking_baseline_repeats"] == 3
    acceptance = step.last_acceptance_record
    assert acceptance["acceptance"]["accepted"]
    assert acceptance["acceptance"]["trials"][-1]["reason"] == "accepted"
    assert acceptance["repeatability"]["repeats"] == 3
    json.dumps(acceptance, allow_nan=False)
    assert_tree_identical((params, state, optimizer_state, args[0]), before)


def test_rejected_batch_retains_params_rms_count_memory_and_rng_exactly():
    _, _, optimizer, params, state, args, step = excess_setup(absolute_floor=1e10)
    optimizer_state = optimizer.init(params)
    before = snapshot((params, state, optimizer_state, args[0]))
    with pytest.raises(JointDescentFailure) as caught:
        step(params, state, optimizer_state, *args)
    assert caught.value.diagnostics["reason"] == "backtracking_exhausted"
    assert len(caught.value.diagnostics["acceptance"]["trials"]) == 9
    assert not caught.value.diagnostics["acceptance"]["accepted"]
    assert_tree_identical((params, state, optimizer_state, args[0]), before)


def test_backtracked_candidates_use_original_params_and_fixed_rms_direction(monkeypatch):
    from src.models.mamba.v24_Ilya.training import mgda_acceptance
    base_learning_rate = 4.
    cfg, _, optimizer, params, state, args, step = excess_setup(learning_rate=base_learning_rate)
    optimizer_state = optimizer.init(params)
    first, second = step.objective_gradients(params, state, args)
    updates, expected_optimizer, _ = optimizer.update_pair(first, second, optimizer_state, params)
    recorded = []
    real_backtrack = mgda_acceptance.backtrack_losses

    def observing_backtrack(before, evaluate_candidate, **kwargs):
        def evaluate(learning_rate):
            values = np.asarray(evaluate_candidate(learning_rate))
            # Independent closed-loop replay at the original parameters plus
            # a scaled displacement, not at the previous rejected candidate.
            candidate = jax.tree_util.tree_map(
                lambda p, u: p + u * (learning_rate / base_learning_rate), params, updates)
            expected = np.asarray(step.forward_only(candidate, state, *args)[0])
            np.testing.assert_allclose(values, expected, rtol=2e-6, atol=2e-6)
            recorded.append((learning_rate, values))
            return values
        return real_backtrack(before, evaluate, **kwargs)

    monkeypatch.setattr(mgda_acceptance, "backtrack_losses", observing_backtrack)
    result = step(params, state, optimizer_state, *args)
    assert len(recorded) >= 2, "Toy full step must overshoot before backtracking succeeds"
    assert result.diagnostics["backtracking_halvings"] == len(recorded) - 1
    assert result.diagnostics["learning_rate"] == recorded[-1][0]
    assert_tree_close(result.optimizer_state, expected_optimizer)
    assert int(result.optimizer_state.count) == 1
    expected_params = jax.tree_util.tree_map(
        lambda p, u: p + u * (recorded[-1][0] / base_learning_rate), params, updates)
    assert_tree_close(result.params, expected_params)
    record = step.last_acceptance_record["acceptance"]
    assert len(record["trials"]) == len(recorded)
    assert not record["trials"][0]["accepted"]
    assert record["trials"][-1]["accepted"]
    assert all("losses" in trial and "learning_rate" in trial and "reason" in trial
               for trial in record["trials"])
    json.dumps(record, allow_nan=False)


def test_all_direction_controls_use_same_pair_rms_moments_for_same_gradients():
    params = {"spatial": jnp.asarray([.2, -.1]), "mesh_interleaved_temporal_0": jnp.asarray([.3, .7])}
    first = jax.tree_util.tree_map(lambda x: 2 * x, params)
    second = jax.tree_util.tree_map(lambda x: -.3 * x, params)
    states = []
    directions = []
    for mode in ("mgda", "mse_control", "fixed_sum"):
        optimizer = RMSOptimizer(lambda count: jnp.asarray(1e-6), direction_mode=mode)
        updates, state, _ = optimizer.update_pair(first, second, optimizer.init(params), params)
        states.append(state)
        directions.append(updates)
    assert_tree_identical(states[0], states[1])
    assert_tree_identical(states[0], states[2])
    expected_moment = jax.tree_util.tree_map(lambda x, y: .02 * .5 * (x ** 2 + y ** 2), first, second)
    assert_tree_close(states[0].second_moment, expected_moment)
    assert not np.array_equal(np.asarray(directions[0]["spatial"]), np.asarray(directions[1]["spatial"]))


def qualification_harness(monkeypatch, failing_mode=None):
    """Expose conspicuously different candidate and unchanged-parent memory."""
    from src.models.mamba.v24_Ilya.training import mgda_step, runner
    from src.models.mamba.v24_Ilya.training.data import TrainingCursor
    from test_mse_amse_excess_loss import excess_config

    config = excess_config(mgda_direction="mse_control")
    params, initial_state = {"w": np.array([.3], np.float32)}, {"s": np.array([5.], np.float32)}
    key = jax.random.PRNGKey(19)
    observed = {"attempts": [], "forwards": [], "cursors": [], "optimizer_states": []}

    class DiscardingOptimizer:
        def init(self, candidate_params):
            assert_tree_identical(candidate_params, params)
            state = {"count": np.array(0), "moment": np.array([0.], np.float32)}
            observed["optimizer_states"].append(state)
            return state

    optimizer = DiscardingOptimizer()
    def build_optimizer(replay_config):
        assert replay_config.mgda_direction == "mgda"
        return optimizer, None
    monkeypatch.setattr(runner, "build_optimizer", build_optimizer)

    def make_step(*, config, optimizer, **kwargs):
        mode = config.feedback_mode
        assert config.mgda_direction == "mgda"
        def step(candidate_params, state, optimizer_state, *args):
            observed["attempts"].append((mode, snapshot(candidate_params), snapshot(state),
                                         snapshot(optimizer_state), snapshot(args[0])))
            if mode == failing_mode:
                raise JointDescentFailure({"reason": "backtracking_exhausted"})
            step.last_acceptance_record = {
                "acceptance": {"accepted": True, "trials": [
                    {"learning_rate": 1e-6, "losses": [.9, 1.9],
                     "accepted": True, "reason": "accepted"}]},
                "repeatability": {"repeats": 3, "repeatability_span": [0., 0.]},
            }
            return SimpleNamespace(
                params={"w": candidate_params["w"] + 100},
                state={"s": state["s"] + 1000},
                optimizer_state={"count": np.array(1), "moment": np.array([9.], np.float32)},
                diagnostics={"finite": np.bool_(True), "objective_pair": "mse_amse_excess",
                             "mgda_direction": "mgda", "objective_1_before": np.float32(1.)})
        def forward_only(candidate_params, state, *args):
            observed["forwards"].append((mode, snapshot(candidate_params), snapshot(state), snapshot(args[0])))
            return np.array([1., 2.]), {"s": state["s"] + 1}, None, None
        step.forward_only = forward_only
        return step
    monkeypatch.setattr(mgda_step, "make_multiobjective_train_step", make_step)

    external_cursor = TrainingCursor()
    def build_chunk(cursor, passed_config, task_config):
        observed["cursors"].append(cursor)
        next_cursor = (TrainingCursor(segment_index=cursor.segment_index, segment_offset=4)
                       if cursor.segment_offset == 0 else TrainingCursor(segment_index=cursor.segment_index + 1))
        return SimpleNamespace(input_frames=(), static_inputs={}, truths=(), forcings=(),
                               raw_anchor_indices=np.array([len(observed["cursors"]) - 1]),
                               next_cursor=next_cursor)
    data = SimpleNamespace(build_chunk=build_chunk, input_steps=2, time_step=pd.Timedelta("6h"),
                           cursor=external_cursor)
    arguments = dict(config=config, transforms=None, training_data=data, task_config=None,
                     params=params, zero_state=initial_state, rng_key=key,
                     baseline_params={}, baseline_state={}, batches=4,
                     provenance={"initialization_checkpoint": "unchanged_parent.pkl"})
    return runner, arguments, observed


def test_four_batch_qualification_discards_both_trials_and_carries_unchanged_parent_memory(monkeypatch):
    runner, arguments, observed = qualification_harness(monkeypatch)
    originals = snapshot((arguments["params"], arguments["zero_state"], arguments["rng_key"]))
    external_cursor = arguments["training_data"].cursor
    result = runner._calibrate_initialized_training(**arguments)
    assert result["passed"] and result["batches"] == 4
    assert not result["parameters_updated"] and result["optimizer_updates"] == 0
    assert len(observed["attempts"]) == 8
    assert len(observed["forwards"]) == 4
    assert [cursor.segment_offset for cursor in observed["cursors"]] == [0, 4, 0, 4]
    assert [cursor.segment_index for cursor in observed["cursors"]] == [0, 0, 1, 1]
    expected_state = [5., 6., 0., 1.]
    key = arguments["rng_key"]
    for index, parent_state in enumerate(expected_state):
        key, *split_keys = jax.random.split(key, arguments["config"].bptt_steps + 1)
        expected_keys = jnp.stack(split_keys)
        for attempt in observed["attempts"][2 * index:2 * index + 2]:
            mode, params, state, rms_state, keys = attempt
            assert_tree_identical(params, arguments["params"])
            np.testing.assert_array_equal(state["s"], [parent_state])
            assert int(rms_state["count"]) == 0
            np.testing.assert_array_equal(keys, expected_keys)
        mode, params, state, keys = observed["forwards"][index]
        assert mode == "closed_loop_sg"
        np.testing.assert_array_equal(state["s"], [parent_state])
        np.testing.assert_array_equal(keys, expected_keys)
        record = result["records"][index]
        for mode in ("baseline", "closed_loop_sg"):
            assert record[mode]["diagnostics"]["finite"] is True
            assert record[mode]["acceptance_record"]["acceptance"]["accepted"]
    assert_tree_identical((arguments["params"], arguments["zero_state"], arguments["rng_key"]), originals)
    assert arguments["training_data"].cursor == external_cursor
    assert all(int(state["count"]) == 0 for state in observed["optimizer_states"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mode", ["baseline", "closed_loop_sg"])
def test_failed_qualification_propagates_without_pass_record_or_parent_state_change(monkeypatch, mode):
    runner, arguments, observed = qualification_harness(monkeypatch, failing_mode=mode)
    originals = snapshot((arguments["params"], arguments["zero_state"], arguments["rng_key"]))
    external_cursor = arguments["training_data"].cursor
    output = None
    with pytest.raises(JointDescentFailure, match="backtracking_exhausted"):
        output = runner._calibrate_initialized_training(**arguments)
    assert output is None
    assert not observed["forwards"], "Rejected qualification cannot advance even parent memory"
    assert_tree_identical((arguments["params"], arguments["zero_state"], arguments["rng_key"]), originals)
    assert arguments["training_data"].cursor == external_cursor
    assert all(int(state["count"]) == 0 for state in observed["optimizer_states"])
