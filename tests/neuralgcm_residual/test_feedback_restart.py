from collections import namedtuple
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.models.neuralgcm_residual.config import RunConfig, BALANCED_LOSS_NAME, FIELDS
from src.models.neuralgcm_residual.corrections import ConstrainedNativeAdapter
from src.models.neuralgcm_residual.loss import balanced_scales, make_weather_loss


def adapter_state():
    core_type = namedtuple('Core', 'vorticity divergence temperature_variation tracers log_surface_pressure sim_time')
    state_type = namedtuple('State', 'state auxiliary')
    grid = SimpleNamespace(modal_shape=(2, 3), nodal_shape=(2, 3),
                           to_modal=lambda x: x, to_nodal=lambda x: x,
                           mask=np.array([[True, True, True], [False, True, True]]))
    model = SimpleNamespace(model_coords=SimpleNamespace(horizontal=grid,
        vertical=SimpleNamespace(centers=[.5])), to_nondim_units=lambda x, u: x,
        from_nondim_units=lambda x, u: x)
    field = jnp.arange(6, dtype=jnp.float32).reshape(1, 2, 3) + 2
    tracers = {k: field for k in FIELDS[-3:]}
    state = state_type(core_type(field, field, field, tracers, field, jnp.float32(12)),
                       {'memory': jnp.array([3., 4.])})
    return ConstrainedNativeAdapter(model, state), state


def test_constraints_preserve_existing_modes_and_carry_not_whole_state_zero():
    adapter, state = adapter_state()
    adapter.assert_zero_identity(state)
    updated = adapter.apply_increment(state, jnp.ones((2, 3, 7)))
    for key in ('divergence', 'vorticity'):
        old, new = getattr(state.state, key), getattr(updated.state, key)
        np.testing.assert_array_equal(new[..., 0], old[..., 0])
        np.testing.assert_array_equal(new[..., 1:], old[..., 1:] + 1)
    np.testing.assert_array_equal(updated.state.log_surface_pressure, state.state.log_surface_pressure)
    np.testing.assert_array_equal(updated.auxiliary['memory'], state.auxiliary['memory'])
    assert updated.state.sim_time == state.state.sim_time
    assert updated.state.temperature_variation[0, 0, 0] == 3


def test_forbidden_increment_gradients_are_zero_and_allowed_modes_train():
    adapter, state = adapter_state()
    def score(delta):
        core = adapter.apply_increment(state, delta).state
        return sum(jnp.sum(x) for x in (core.divergence, core.vorticity, core.log_surface_pressure))
    grad = jax.grad(score)(jnp.ones((2, 3, 7)))
    np.testing.assert_array_equal(grad[..., -1], 0)
    np.testing.assert_array_equal(grad[:, 0, :2], 0)
    np.testing.assert_array_equal(grad[:, 1:, :2], 1)


def test_balanced_loss_pools_cloud_floor_and_squares_amplitude_factor():
    scales = {k: [1., 3.] for k in FIELDS}
    scales[FIELDS[-1]] = [1e-9, 1e-5]
    pooled = balanced_scales(scales)
    expected = np.sqrt((1e-18 + 1e-10) / 2) / .05
    np.testing.assert_allclose(pooled[FIELDS[-1]], [expected, expected])
    np.testing.assert_allclose(pooled['specific_humidity'], np.array([1., 3.]) / .66)
    latitude = np.arcsin(np.polynomial.legendre.leggauss(2)[0])
    loss = make_weather_loss(BALANCED_LOSS_NAME, latitude, [100., 900.], scales)
    truth = {k: jnp.zeros((2, 2, 2)) for k in FIELDS}
    prediction = dict(truth, specific_cloud_liquid_water_content=jnp.full((2, 2, 2), 1e-5))
    score = float(loss.field_scores(prediction, truth)[FIELDS[-1]])
    assert score == pytest.approx((1e-5 / expected) ** 2, rel=1e-6)
    assert loss.name == BALANCED_LOSS_NAME


def test_config_versions_feedback_and_loss_without_changing_legacy_identity():
    original = RunConfig()
    assert 'correction_policy' not in original.to_dict()
    new = RunConfig(correction_policy='no_pressure_zero_mean_v2', loss=BALANCED_LOSS_NAME)
    assert original.identity != new.identity
    with pytest.raises(ValueError):
        RunConfig(correction_policy='unknown')


def test_nonfinite_validation_is_recorded_without_selecting_or_aborting(tmp_path, monkeypatch):
    from src.models.neuralgcm_residual import worker, evaluate
    checkpoint = tmp_path / 'checkpoint.pkl'
    checkpoint.write_bytes(b'smoke checkpoint identity')
    monkeypatch.setattr(worker, 'CacheReader', lambda *a: None)
    monkeypatch.setattr(worker, 'read_json', lambda p: {'origins': ['2022-01-30T00']})
    calls = []
    def failed(*args, warm):
        calls.append(warm)
        return {'eligible_for_selection': False, 'failures': [{'error': 'nonfinite'}]}
    monkeypatch.setattr(evaluate, 'evaluate_origins', failed)
    monkeypatch.setattr(evaluate, 'select_checkpoint', lambda *a, **k: pytest.fail('Selected invalid checkpoint'))
    callback = worker.validation_callback(tmp_path, {}, SimpleNamespace(store=None),
        {'validation_origins': 'unused', 'cache_root': 'unused'}, tmp_path / 'fine', 'finetune')
    callback({}, checkpoint, 5)
    assert calls == [False, True]
    assert (tmp_path / 'fine/validation/000005/selection_rejected.json').is_file()
    assert (tmp_path / 'fine/validation/000005/COMPLETE.json').is_file()


def test_k2_episode_feeds_corrected_first_state_and_targets_exact_leads():
    from src.models.neuralgcm_residual.finetune import episode_gradients
    advances, targets = [], []
    class Backbone:
        model = None
        def encode(self, *args): return jnp.float32(0)
        def advance_6h(self, state, forcing):
            advances.append(float(state))
            return state + 1
        def assert_time(self, initial, state, lead): pass
    class Store:
        def inputs_and_forcing(self, *args): return {}, {}
        def frame(self, target):
            targets.append(str(target))
            return jnp.float32(1)
    trainer = SimpleNamespace(record=lambda *args: args,
        forward=lambda p, h, k, old, base, forcing, target, known: (base * .1, h + 1, base + .5),
        reverse=lambda p, tape: {'length': len(tape)})
    result = episode_gradients(trainer, {}, jnp.float32(0), jax.random.PRNGKey(22),
        backbone=Backbone(), store=Store(), known=lambda *a: jnp.float32(0),
        origin='2022-01-30T00', rollout_steps=2)
    assert advances == [0., 1.5]
    assert targets == ['2022-01-30T06', '2022-01-30T12']
    assert result['valid_hours'] == [6, 12]
    assert result['gradients']['length'] == 2
    assert float(result['physical']) == 3
