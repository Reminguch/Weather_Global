from collections import namedtuple
from dataclasses import replace
from types import SimpleNamespace
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.models.neuralgcm_residual.config import RunConfig, Architecture, initial_matrix
from src.models.neuralgcm_residual.data import complete_segments, eligible_origins, seasonal_order
from src.models.neuralgcm_residual.data import require_complete_experiment_data
from src.models.neuralgcm_residual.loss import WeatherLoss, aggregate_reduction, gaussian_weights
from src.models.neuralgcm_residual.native_state import NativeAdapter
from src.models.neuralgcm_residual.checkpoint import save_checkpoint, load_checkpoint, transfer_parent


def test_matrix_and_locked_defaults():
    matrix = initial_matrix()
    assert len({c.run_id for c in matrix}) == 8
    assert {c.resolution for c in matrix} == {2.8, 1.4}
    assert all(c.architecture.dt_rank == 32 and c.architecture.msg_steps == 2 for c in matrix)
    assert all(c.train_years == tuple(range(2015, 2022)) for c in matrix)
    assert all((c.validation_year, c.test_year) == (2022, 2023) for c in matrix)
    with pytest.raises(ValueError):
        RunConfig(rollout_steps=21)
    with pytest.raises(ValueError):
        Architecture(dt_rank="auto")


def test_segments_drop_gaps_and_boundaries():
    t = np.arange(np.datetime64('2020-01-01'), np.datetime64('2020-02-01'), np.timedelta64(6, 'h'))
    t = np.delete(t, 100)
    segments, dropped = complete_segments(t, 24)
    for indices in segments:
        assert np.all(np.diff(t[indices]) == np.timedelta64(6, 'h'))
    assert len({i for s in segments for i in s} | set(dropped)) == len(t)
    assert 100 in dropped or all(100 not in s or 99 not in s for s in segments)


def test_production_rejects_incomplete_seven_year_training_data():
    times = np.arange(np.datetime64('2015-01-01T00'), np.datetime64('2024-01-01T00'), np.timedelta64(6,'h'))
    assert len(times) == 13148
    require_complete_experiment_data(SimpleNamespace(times=times))
    with pytest.raises(ValueError, match='missing=1'):
        require_complete_experiment_data(SimpleNamespace(times=np.delete(times,1234)))


def test_origin_causality_and_seasons():
    times = np.arange(np.datetime64('2021-12-01'), np.datetime64('2023-02-01'), np.timedelta64(6, 'h'))
    origins = eligible_origins(times, 'val', 40)
    assert min(origins) == '2022-01-03T00'
    assert max(origins) == '2022-12-21T00'
    order = seasonal_order(origins)
    assert order == seasonal_order(origins)
    assert len({(int(t[5:7]) % 12) // 3 for t in order[:4]}) == 4


def test_named_loss_is_gaussian_and_aggregate_mse():
    from src.models.neuralgcm_residual.config import FIELDS
    lat = np.arcsin(np.polynomial.legendre.leggauss(4)[0])
    loss = WeatherLoss(lat, [100, 900], {k: [1, 1] for k in FIELDS})
    target = {k: jnp.zeros((2, 8, 4)) for k in FIELDS}
    prediction = {k: jnp.ones((2, 8, 4)) for k in FIELDS}
    assert float(loss(prediction, target)) == pytest.approx(1)
    assert aggregate_reduction([1, 9], [2, 18]) == 50
    with pytest.raises(ValueError):
        gaussian_weights(np.linspace(-np.pi/2, np.pi/2, 4))


Core = namedtuple('Core', 'vorticity divergence temperature_variation tracers log_surface_pressure sim_time')
State = namedtuple('State', 'state auxiliary')


def test_native_increment_preserves_all_carry_and_zero_baseline():
    grid = SimpleNamespace(modal_shape=(2, 2), nodal_shape=(2, 2),
                           to_nodal=lambda x: x, to_modal=lambda x: x,
                           mask=np.array([[True, False], [True, True]]))
    model = SimpleNamespace(model_coords=SimpleNamespace(horizontal=grid, vertical=SimpleNamespace(centers=[0.5])),
                            from_nondim_units=lambda x, u: x * (10 if u == 'K' else 1),
                            to_nondim_units=lambda x, u: x / (10 if u == 'K' else 1))
    field = jnp.arange(4, dtype=jnp.float32).reshape(1, 2, 2)
    tracers = {k: field for k in ('specific_humidity','specific_cloud_ice_water_content','specific_cloud_liquid_water_content')}
    state = State(Core(field, field, field, tracers, field, jnp.float32(12)), {'keep':jnp.array([3.,4.])})
    adapter = NativeAdapter(model, state)
    adapter.assert_zero_identity(state)
    updated = adapter.apply_increment(state, jnp.ones((2, 2, 7)))
    assert updated.state.sim_time == state.state.sim_time
    np.testing.assert_array_equal(updated.auxiliary['keep'], state.auxiliary['keep'])
    assert updated.state.temperature_variation[0,0,0] == pytest.approx(0.1)
    assert updated.state.temperature_variation[0,0,1] == 1  # masked increment, original preserved
    grad = jax.grad(lambda d: jnp.sum(adapter.apply_increment(state,d).state.temperature_variation))(jnp.zeros((2,2,7)))
    assert float(grad[...,2].sum()) == pytest.approx(.3)


def test_checkpoint_exact_resume_and_transfer_are_distinct(tmp_path):
    ids={'architecture':'a','backbone':'b','native_schema':'c','normalization':'d','dataset':'e'}
    path=tmp_path/'state.pkl'
    memory={'h':jnp.array([2.])}
    save_checkpoint(path,stage='pretrain',identities=ids,params={'w':jnp.array([3.])},
                    optimizer={'step':7},memory=memory,rng=jax.random.PRNGKey(22),
                    cursor={'epoch':3,'segment':5,'chunk':2,'update':50})
    state=load_checkpoint(path,stage='pretrain',identities=ids)
    assert state['cursor']['chunk']==2 and state['optimizer']['step']==7
    np.testing.assert_array_equal(state['memory']['h'],memory['h'])
    params,parent=transfer_parent(path,ids)
    assert set(params)=={'w'} and 'optimizer' in parent['reset']
    with pytest.raises(ValueError):
        load_checkpoint(path,stage='finetune',identities=ids)
    with pytest.raises(ValueError):
        load_checkpoint(path,stage='pretrain',identities=dict(ids,dataset='changed'))
