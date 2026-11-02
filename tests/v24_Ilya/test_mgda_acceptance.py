"""Finite-step safeguards on deterministic, noisy and rounded toy objectives."""
from dataclasses import FrozenInstanceError
from functools import partial
import json

import numpy as np
import pytest

from src.models.mamba.v24_Ilya.training.mgda_acceptance import (
    LossTolerances, backtrack_losses as _backtrack_losses, calibrate_loss_tolerances,
)

# Historical strict behavior remains covered explicitly; production defaults
# to advisory scheduled updates.
backtrack_losses = partial(_backtrack_losses, decrease_policy="error")


def tolerances(losses=(1., 2.), **kwargs):
    return calibrate_loss_tolerances([losses] * 3, **kwargs)


def test_backtracking_repairs_finite_step_overshoot_without_advancing_input_state():
    params = np.array([0.])
    memory = {"value": np.array([7.])}
    rng = np.array([1, 23], dtype=np.uint32)
    snapshots = [params.copy(), memory["value"].copy(), rng.copy()]
    trials = []
    # Both exact gradients point toward larger x. A large finite step still
    # increases the first loss; replaying a smaller step improves both.
    def evaluate(learning_rate):
        candidate = params + learning_rate * 6e6
        trial_memory = memory["value"].copy() + candidate
        trial_rng = rng.copy()
        trials.append((candidate.copy(), trial_memory, trial_rng))
        return float((candidate[0] - 2) ** 2), float((candidate[0] - 3) ** 2)

    result = backtrack_losses((4., 9.), evaluate, mode="mgda", tolerances=tolerances((4., 9.)))
    assert result.accepted
    assert result.learning_rate == 5e-7
    assert result.losses == (1., 0.)
    assert len(result.trials) == 2
    assert not result.trials[0].accepted
    np.testing.assert_array_equal(trials[0][0], [6.])
    np.testing.assert_array_equal(trials[1][0], [3.])
    for actual, original in zip((params, memory["value"], rng), snapshots, strict=True):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("mode, accepted", [("mgda", False), ("mse_control", True), ("fixed_sum", True)])
def test_controls_enforce_their_own_objective_instead_of_common_descent(mode, accepted):
    result = backtrack_losses((1., 2.), lambda lr: (.5, 2.25), mode=mode,
                             tolerances=tolerances(), max_halvings=0)
    assert result.accepted == accepted


def test_fixed_sum_rejects_sum_increase_even_when_mse_improves():
    result = backtrack_losses((1., 2.), lambda lr: (.5, 3.), mode="fixed_sum",
                             tolerances=tolerances(), max_halvings=0)
    assert not result.accepted


@pytest.mark.parametrize("mode", ["mgda", "mse_control", "fixed_sum"])
@pytest.mark.parametrize("candidate", [(1.01, 2.01), (1., 2.)])
def test_advisory_default_commits_finite_increase_or_tie_at_scheduled_lr(mode, candidate):
    calls = []
    def evaluate(lr):
        calls.append(lr)
        return candidate
    with pytest.warns(RuntimeWarning, match="committing finite scheduled update"):
        result = _backtrack_losses((1., 2.), evaluate, mode=mode, tolerances=tolerances())
    assert calls == [1e-6]
    assert result.accepted and not result.measured_decrease
    assert result.reason == "advisory_no_decrease" and result.decrease_policy == "warn"
    assert result.losses == candidate and result.learning_rate == 1e-6
    assert not result.trials[0].measured_decrease
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("candidate", [(np.nan, 2.), (1., np.inf)])
def test_advisory_default_never_commits_nonfinite_candidate(candidate):
    result = _backtrack_losses((1., 2.), lambda lr: candidate, mode="mgda", tolerances=tolerances())
    assert not result.accepted and result.reason == "nonfinite_loss"
    assert len(result.trials) == 1 and result.losses is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_identical_replay_noise_sets_strict_decrease_floor():
    # Powers of two make exact equality at the acceptance boundary testable.
    noise = 2. ** -20
    qualified = calibrate_loss_tolerances([(1., 2.), (1. + noise, 2. + noise), (1., 2.)],
                                         absolute_floor=0, relative_floor=0, repeatability_policy="error")
    assert qualified.repeatability_span == (noise, noise)
    assert qualified.thresholds((1., 2.)) == (4 * noise, 4 * noise)
    calls = []
    def evaluate(lr):
        calls.append(lr)
        decrease = 4 * noise if len(calls) == 1 else 5 * noise
        return 1. - decrease, 2. - decrease
    result = backtrack_losses((1., 2.), evaluate, mode="mgda", tolerances=qualified)
    assert len(result.trials) == 2
    assert not result.trials[0].accepted
    assert result.accepted
    assert result.trials[1].decreases == (5 * noise, 5 * noise)
    json.dumps(qualified.to_dict(), allow_nan=False)


def test_explicit_legacy_strict_policy_rejects_noise():
    with pytest.raises(ValueError, match="configured bound"):
        calibrate_loss_tolerances([(1., 2.), (1.01, 2.), (1., 2.)],
                                 max_relative_noise=1e-3, repeatability_policy="error")
    # At zero, the explicit absolute floor bounds the permissible variation.
    with pytest.raises(ValueError, match="configured bound"):
        calibrate_loss_tolerances([(0., 0.), (1e-7, 0.), (0., 0.)],
                                 absolute_floor=1e-8, repeatability_policy="error")


@pytest.mark.parametrize("mode", ["mgda", "mse_control", "fixed_sum"])
def test_actual_failed_pilot_variance_warns_and_does_not_indirectly_reject_decrease(mode):
    repeated = [[3.5118603706359863, .9506522417068481],
                [3.504420042037964, .9496756792068481],
                [3.510044813156128, .9508850574493408]]
    with pytest.warns(RuntimeWarning, match="advisory only"):
        qualified = calibrate_loss_tolerances(repeated)
    before = np.mean(repeated, axis=0)
    assert qualified.noise_exceeded == (True, True)
    assert qualified.repeatability_policy == "warn"
    np.testing.assert_allclose(qualified.thresholds(before), before * 1e-7)
    # A measured decrease smaller than repeatability noise is allowed, with
    # the larger noise-aware margin retained for interpretation.
    result = backtrack_losses(before, lambda lr: before - 1e-4, mode=mode, tolerances=qualified)
    assert result.accepted
    assert all(floor > 1e-4 for floor in qualified.noise_thresholds(before))
    json.dumps(qualified.to_dict(), allow_nan=False)
    worse = backtrack_losses(before, lambda lr: before + 1e-4, mode=mode,
                            tolerances=qualified, max_halvings=0)
    assert not worse.accepted


def test_warning_policy_still_records_near_zero_noise_and_rejects_invalid_policy():
    with pytest.warns(RuntimeWarning):
        qualified = calibrate_loss_tolerances([(0., 0.), (1e-7, 0.), (0., 0.)])
    assert qualified.noise_exceeded == (True, False)
    with pytest.raises(ValueError, match="repeatability_policy"):
        tolerances(repeatability_policy="ignore")


def test_zero_objectives_and_stationary_direction_stall_without_accepting_ties():
    before = np.array([0., 0.])
    returned = before.copy()
    calls = []
    def evaluate(lr):
        calls.append(lr)
        return returned
    result = backtrack_losses(before, evaluate, mode="mgda", tolerances=tolerances((0., 0.)))
    assert not result.accepted
    assert result.learning_rate is None and result.losses is None
    assert result.reason == "backtracking_exhausted"
    assert len(calls) == 9
    assert calls[0] == 1e-6 and calls[-1] == 1e-6 / 256
    assert all(not trial.accepted for trial in result.trials)
    np.testing.assert_array_equal(before, [0., 0.])
    np.testing.assert_array_equal(returned, before)


def test_parameter_rounding_cannot_be_reported_as_measured_improvement():
    parameter = np.float32(1e8)
    baseline = (float(parameter ** 2), float(parameter ** 2))
    def evaluate(lr):
        candidate = np.float32(parameter - np.float32(lr))
        return float(candidate ** 2), float(candidate ** 2)
    result = backtrack_losses(baseline, evaluate, mode="mgda", tolerances=tolerances(baseline))
    assert not result.accepted
    assert all(trial.decreases == (0., 0.) for trial in result.trials)


def test_nonfinite_candidate_is_recorded_and_backtracked_with_portable_json():
    calls = []
    def evaluate(lr):
        calls.append(lr)
        return (np.nan, np.inf) if len(calls) == 1 else (.9, 1.9)
    result = backtrack_losses((1., 2.), evaluate, mode="mgda", tolerances=tolerances())
    assert result.accepted
    assert result.trials[0].reason == "nonfinite_loss"
    record = result.to_dict()
    assert record["trials"][0]["losses"] == [None, None]
    json.dumps(record, allow_nan=False)


def test_model_exceptions_are_not_hidden_as_optimizer_stalls():
    def evaluate(lr):
        raise RuntimeError("broken forward")
    with pytest.raises(RuntimeError, match="broken forward"):
        backtrack_losses((1., 2.), evaluate, mode="mgda", tolerances=tolerances())


@pytest.mark.parametrize("values", [[], [(1., 2.)] * 2, [(1.,)] * 3, [(np.nan, 2.)] * 3])
def test_repeatability_requires_three_finite_identical_shape_forward_records(values):
    with pytest.raises(ValueError):
        calibrate_loss_tolerances(values)


@pytest.mark.parametrize("kwargs", [
    {"initial_learning_rate": 0}, {"initial_learning_rate": np.inf},
    {"max_halvings": 9}, {"max_halvings": -1}, {"max_halvings": 1.5},
    {"shrink": .1}, {"mode": "mse"},
])
def test_invalid_policy_fails_before_model_evaluation(kwargs):
    def evaluate(lr):
        raise AssertionError("Invalid policy must fail before calling the model")
    arguments = {"mode": "mgda", "tolerances": tolerances(), **kwargs}
    with pytest.raises(ValueError):
        backtrack_losses((1., 2.), evaluate, **arguments)


def test_repeatability_record_is_immutable_and_validates_direct_construction():
    qualified = tolerances()
    with pytest.raises(FrozenInstanceError):
        qualified.repeats = 10
    record = qualified.to_dict()
    record["absolute_floor"][0] = 100
    assert qualified.absolute_floor[0] == 1e-8
    assert LossTolerances(**qualified.to_dict()) == qualified
    with pytest.raises(ValueError, match="nonnegative"):
        calibrate_loss_tolerances([(1., 2.)] * 3, relative_floor=-1)
