"""Nonnegative spectral adjustment and exact-grid accuracy objective contract."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

from graphcast import losses as gc_losses
from graphcast import xarray_jax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.models.mamba.v24_Ilya.metrics import ORIGINAL_GRAPHCAST_VARIABLE_WEIGHTS
from src.models.mamba.v24_Ilya.training.config import (
    MGDA_BACKTRACKING_DEFAULTS, load_training_config, validate_resume_config,
)
from src.models.mamba.v24_Ilya.training.mse_amse_excess_loss import (
    build_mse_amse_excess_objectives,
)
from src.models.mamba.v24_Ilya.training.spectral_loss import (
    amse_per_degree, build_full_field_loss,
)


@pytest.mark.parametrize("epsilon", [1e-12, .02])
@pytest.mark.parametrize("coefficient", [0., .3, 1.])
def test_direct_excess_matches_regularized_amse_minus_spectral_mse_and_gradient(epsilon, coefficient):
    truth = jnp.asarray([[1., .2j], [.3, .1-.2j]], jnp.complex64)
    prediction = jnp.asarray([[.45, .15+.5j], [.1, -.2j]], jnp.complex64)

    def objectives(scale):
        amse, terms = amse_per_degree(prediction * scale, truth, epsilon, coefficient)
        return jnp.stack((terms["amse_excess"].sum(), (amse - terms["spectral_mse"]).sum()))

    value, gradient = jax.jit(lambda s: (objectives(s), jax.jacrev(objectives)(s)))(.83)
    assert np.all(np.asarray(value[0]) >= 0)
    np.testing.assert_allclose(value[0], value[1], rtol=2e-5, atol=5e-7)
    np.testing.assert_allclose(gradient[0], gradient[1], rtol=2e-5, atol=5e-7)


@pytest.mark.parametrize("amplitude", [0., 1e-10, 1., 1e4])
def test_excess_nonnegative_finite_and_zero_for_identical_coefficients(amplitude):
    truth = jnp.asarray([[amplitude, amplitude * .25j]], jnp.complex64)
    loss = lambda prediction: amse_per_degree(prediction, truth)[1]["amse_excess"].sum()
    value, gradient = jax.value_and_grad(loss)(truth)
    assert float(value) == 0
    np.testing.assert_array_equal(gradient, 0)
    for factor in (0., .2, -.2, 2.):
        value, gradient = jax.value_and_grad(loss)(truth * factor)
        assert np.isfinite(value) and float(value) >= 0
        assert np.isfinite(gradient).all()


def test_equal_power_incoherent_fields_have_zero_excess_and_symmetric_subgradient():
    truth = jnp.asarray([[1., .3j]], jnp.complex64)
    prediction = jnp.asarray([[1., -.3j]], jnp.complex64)
    loss = lambda scale: amse_per_degree(prediction * scale, truth)[1]["amse_excess"].sum()
    value, gradient = jax.value_and_grad(loss)(1.)
    assert float(value) == 0
    assert float(gradient) == 0
    assert float(loss(.99)) > 0 and float(loss(1.01)) > 0


def test_direct_excess_keeps_small_positive_adjustment_next_to_large_mse():
    prediction = jnp.asarray([[10000.001]], jnp.complex64)
    truth = jnp.asarray([[-10000.]], jnp.complex64)
    _, terms = amse_per_degree(prediction, truth)
    # Opposite real coefficients have decorrelation=4 up to epsilon. Use the
    # actual FP32 coefficient values to make the reference independent of decimal
    # input rounding; avoid subtracting two O(1e8) losses to recover O(1e2).
    a = float(np.asarray(prediction).real[0, 0])
    b = abs(float(np.asarray(truth).real[0, 0]))
    expected = 4 * a * (a - b)
    assert expected > 0
    np.testing.assert_allclose(terms["amse_excess"], [expected], rtol=2e-6)


def test_full_83_channel_pair_uses_exact_gc_weights_and_shared_spectral_normalization():
    from test_amse_split_mgda_core import fields83

    truth, stats = fields83()
    prediction = truth * .73 + .1
    amse = build_full_field_loss(stats, "amse", lmax=5)
    pair = build_mse_amse_excess_objectives(amse)
    values, diagnostic = pair(prediction, truth)
    expected_gc, _ = gc_losses.weighted_mse_per_level(
        prediction / stats, truth / stats, ORIGINAL_GRAPHCAST_VARIABLE_WEIGHTS)
    np.testing.assert_allclose(values[0], xarray_jax.unwrap_data(expected_gc), rtol=2e-6)
    total, spectral = amse(prediction, truth)
    np.testing.assert_allclose(values[1], total - spectral["spectral_mse"], rtol=2e-5, atol=2e-6)
    assert np.all(np.asarray(values) >= 0)
    assert values.shape == (2, 2) and values.dtype == jnp.float32
    assert set(diagnostic) == {"gc_loss", "amse_excess", "spectral_mse", "amse_loss"}
    gradient = jax.jacrev(lambda scale: pair(truth * scale + .1, truth)[0].sum(axis=1))(.73)
    assert gradient.shape == (2,) and np.isfinite(gradient).all()
    # Vertical velocity was omitted by the older nine-field power objective.
    changed = prediction.copy()
    changed["vertical_velocity"] = truth["vertical_velocity"] * -10
    assert np.all(np.abs(np.asarray(pair(changed, truth)[0] - values)).max(axis=1) > 1)


def excess_config(**changes):
    from test_amse_split_mgda_core import config

    return replace(config(root=Path("excess_test")), objective_pair="mse_amse_excess",
                   optimizer_kind="mgda_rms", mgda_backtracking={}, **changes)


def test_training_transform_routes_new_pair_to_exact_gc_and_excess():
    from test_amse_split_mgda_core import fields83
    from src.models.mamba.v24_Ilya.training.endpoint_step import build_training_transforms

    truth, stats = fields83()
    cfg = excess_config(amse_lmax=5)
    transforms = build_training_transforms(
        SimpleNamespace(baseline=None, residual=None), None, stats, cfg)
    assert transforms.full_field_objectives is not None
    assert transforms.residual_supervised_predict is not None
    prediction = truth * .7 + .1
    actual, diagnostic = transforms.full_field_objectives(prediction, truth)
    _, expected = transforms.full_field_loss(prediction, truth)
    np.testing.assert_array_equal(actual[0], expected["grid_mse_fp32"])
    np.testing.assert_array_equal(actual[1], expected["amse_excess"])
    np.testing.assert_array_equal(diagnostic["gc_loss"], actual[0])


@pytest.mark.parametrize("direction", ["mgda", "mse_control", "fixed_sum"])
def test_new_pair_config_roundtrip_freezes_direction_and_step_acceptance(tmp_path, direction):
    cfg = excess_config(mgda_direction=direction)
    assert cfg.mgda_backtracking == MGDA_BACKTRACKING_DEFAULTS
    assert cfg.power_calibration is None and cfg.amse_split_calibration is None
    assert cfg.to_dict()["optimizer"]["mgda_direction"] == direction
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert load_training_config(path).to_dict() == cfg.to_dict()
    assert replace(cfg, feedback_mode="baseline").feedback_mode == "baseline"


@pytest.mark.parametrize("direction", ["mgda", "mse_control", "fixed_sum"])
def test_full_feedback_config_preserves_window_and_rejects_changed_resume(tmp_path, direction):
    cfg = excess_config(mgda_direction=direction, feedback_mode="closed_loop",
                        bptt_steps=24, ar_tail_k=20, segment_steps=96)
    assert cfg.truth_prefix_steps == 4
    assert cfg.bptt_backend == "explicit_weather_reverse_vjp"
    assert cfg.normalized_supervised_weights == (1 / 24,) * 24
    path = tmp_path / "full_feedback.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert load_training_config(path) == cfg
    validate_resume_config(cfg, cfg.to_dict(), completed_step=8)
    stopped = replace(cfg, feedback_mode="closed_loop_sg")
    with pytest.raises(ValueError, match="Resume configuration differs"):
        validate_resume_config(cfg, stopped.to_dict(), completed_step=8)
    with pytest.raises(ValueError, match="divisible"):
        replace(cfg, segment_steps=100)
    with pytest.raises(ValueError, match="ar_tail_k"):
        replace(cfg, ar_tail_k=24)


def test_pair_memory_contract_counts_two_sequential_weather_reverse_sweeps():
    from src.models.mamba.v24_Ilya.training.endpoint_step import memory_contract
    cfg = excess_config(feedback_mode="closed_loop", bptt_steps=24, ar_tail_k=20, segment_steps=96)
    contract = memory_contract(input_frames=({"x": np.zeros((4, 2), np.float32)},) * 5,
        static_inputs={}, truths=({"x": np.zeros((4, 2), np.float32)},) * 24,
        residual_state={"s": np.zeros((4, 2), np.float32)}, config=cfg)
    assert contract["graphcast_backward_calls"] == 48
    assert contract["objective_reverse_sweeps"] == 2
    assert contract["concurrent_weather_cotangent_tapes"] == 1
    assert contract["teacher_input_windows"] == 4
    assert contract["gradient_window_steps"] == contract["weather_restart_every_steps"] == 24
    assert contract["recurrent_gradient_truncation_steps"] == 24
    assert contract["graphcast_input_gradients"] and not contract["graphcast_parameters_trainable"]


@pytest.mark.parametrize("changes,match", [
    ({"mgda_backtracking": None}, "post-update backtracking"),
    ({"mgda_backtracking": {"max_halvings": -1}}, "max_halvings"),
    ({"mgda_backtracking": {"max_halvings": 9}}, "max_halvings"),
    ({"mgda_backtracking": {"shrink": 1.}}, "must equal 0.5"),
    ({"mgda_backtracking": {"relative_floor": float("nan")}}, "relative_floor"),
    ({"mgda_backtracking": {"absolute_floor": True}}, "absolute_floor"),
    ({"mgda_backtracking": {"repeatability_repeats": 1}}, "repeatability_repeats"),
    ({"mgda_backtracking": {"repeatability_repeats": 2}}, "repeatability_repeats"),
    ({"mgda_backtracking": {"max_relative_noise": 1.}}, "max_relative_noise"),
    ({"mgda_backtracking": {"repeatability_policy": "ignore"}}, "repeatability_policy"),
    ({"mgda_backtracking": {"decrease_policy": "ignore"}}, "decrease_policy"),
    ({"mgda_backtracking": {"unknown": 1}}, "Unknown"),
    ({"mgda_direction": "other"}, "mgda_direction"),
    ({"amse_lambda": .5}, "unmodified lambda=1"),
    ({"optimizer_kind": "adamw"}, "requires mgda_rms"),
])
def test_new_pair_config_rejects_unprotected_or_changed_objectives(changes, match):
    cfg = excess_config()
    with pytest.raises(ValueError, match=match):
        replace(cfg, **changes)


def test_legacy_pair_schema_unchanged_and_controls_cannot_leak_into_old_runs():
    from test_amse_split_mgda_core import config

    legacy = config()
    assert "mgda_direction" not in legacy.to_dict()["optimizer"]
    assert "mgda_backtracking" not in legacy.to_dict()["optimizer"]
    with pytest.raises(ValueError, match="require objective_pair"):
        replace(legacy, mgda_backtracking={})
