"""Numerical contracts, controller pathologies, and restart tests (CPU)."""
from types import SimpleNamespace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experimental.adaptive_loss.controller import BalanceConfig, WindowBalancer, bounded_mean_one
from experimental.adaptive_loss.objective import GroupedLoss, calibrate
from src.models.neuralgcm_residual.config import FIELDS
from src.models.neuralgcm_residual.native_state import NATIVE_FIELDS
from src.models.neuralgcm_residual.pressure_weighted_loss import PressureWeightedLoss


def fixture(steps=2):
    grid=SimpleNamespace(modal_shape=(2,3),mask=np.array([[1,1,1],[0,1,1]]),radius=1.)
    model=SimpleNamespace(data_coords=SimpleNamespace(horizontal=grid,
        vertical=SimpleNamespace(centers=np.array([10.,30.,500.]))),
        model_coords=SimpleNamespace(horizontal=grid))
    fields={'data':FIELDS,'model':tuple(k for k,_ in NATIVE_FIELDS)}
    stats={'split':'train','lag_hours':24,
           'scales':{s:{k:[1.] for k in names} for s,names in fields.items()}}
    rng=np.random.default_rng(1)
    def tree():
        return {s:{k:jnp.asarray(rng.normal(size=(2,steps,3 if s=='data' else 2,2,3)),jnp.float32)
                   for k in names} for s,names in fields.items()}
    return model,stats,tree(),tree()


@pytest.mark.parametrize('steps',[1,2,20])
def test_decomposition_and_parameter_gradient_equal_fixed_objective(steps):
    m,s,p,t=fixture(steps)
    fixed=PressureWeightedLoss(m,s,steps,upper_pressure_weight=1e-5)
    loss=GroupedLoss(m,s,steps)
    coefficients=jnp.ones(len(loss.names))
    value,gradient=jax.value_and_grad(lambda x:loss.weighted(x,t,coefficients)[0])(p)
    expected,reference=jax.value_and_grad(fixed)(p,t)
    np.testing.assert_allclose(value,expected,rtol=3e-6)
    for a,b in zip(jax.tree_util.tree_leaves(gradient),jax.tree_util.tree_leaves(reference),strict=True):
        np.testing.assert_allclose(a,b,rtol=3e-5,atol=2e-6)
    np.testing.assert_allclose(sum(loss.priors),1,rtol=1e-6)


def test_dynamic_weights_survive_jit_and_are_detached():
    m,s,p,t=fixture()
    loss=GroupedLoss(m,s,2)
    f=jax.jit(jax.value_and_grad(loss.weighted,has_aux=True))
    w=jnp.ones(len(loss.names))
    (a,_),ga=f(p,t,w)
    (b,_),gb=f(p,t,w*2)
    np.testing.assert_allclose(b,2*a,rtol=2e-6)
    for x,y in zip(jax.tree_util.tree_leaves(ga),jax.tree_util.tree_leaves(gb),strict=True):
        np.testing.assert_allclose(y,2*x,rtol=2e-6,atol=1e-6)
    np.testing.assert_array_equal(jax.grad(lambda z:loss.weighted(p,t,z)[0])(w),0)


def test_upper_coefficient_does_not_reweight_weather_band():
    m,s,p,t=fixture()
    a,b=GroupedLoss(m,s,2,beta=1),GroupedLoss(m,s,2,beta=1e-5)
    x,y=np.asarray(a.components(p,t)),np.asarray(b.components(p,t))
    for i,n in enumerate(a.names):
        np.testing.assert_allclose(y[i],x[i]*(1e-5 if n.endswith('/upper') else 1),rtol=2e-6)


def test_controller_reduces_gradient_dominance_without_erasing_a_group():
    c=WindowBalancer(['large','small','pressure'],BalanceConfig(window=5,interval=2),excluded=['pressure'])
    for step in range(100):
        old=c.weights.copy()
        c.observe([1.,1.,9999.], [100.,1.,0.] if c.needs_probe() else None)
        assert np.all(c.weights>=.25-1e-10) and np.all(c.weights<=4+1e-10)
        assert c.weights[2]==1
        np.testing.assert_allclose(c.weights[:2].mean(),1,atol=1e-12)
        assert np.all(c.weights<=old*1.2+1e-10) and np.all(c.weights>=old*.8-1e-10)
    assert c.weights[0] < c.weights[1]
    # Equalization is not guaranteed if the box prevents sufficient correction.
    assert c.weights[0]*100 > c.weights[1]


def test_slow_learning_receives_more_weight_at_equal_gradient_norm():
    c=WindowBalancer(['improving','stalled'],BalanceConfig(window=3,interval=1))
    c.observe([1.,1.],[1.,1.])
    for _ in range(5):c.observe([.2,1.],[1.,1.])
    assert c.weights[1]>c.weights[0]


def test_controller_exact_json_restart_between_probe_boundaries():
    a=WindowBalancer(['a','b'],BalanceConfig(window=7,interval=3))
    for i in range(5):a.observe([1/(i+1),2.],[2.,1.] if a.needs_probe() else None)
    b=WindowBalancer.from_state(json.loads(json.dumps(a.state_dict())))
    for i in range(5,20):
        for c in (a,b):c.observe([1/(i+1),2.],[2.,1.] if c.needs_probe() else None)
        assert a.state_dict()==b.state_dict()


@pytest.mark.parametrize('which',['loss','gradient'])
def test_nonfinite_does_not_change_controller_state(which):
    c=WindowBalancer(['a','b'],BalanceConfig(interval=1))
    c.observe([1.,2.],[1.,2.]);before=c.state_dict()
    with pytest.raises(FloatingPointError):
        c.observe([np.nan,2.] if which=='loss' else [1.,2.],
                  [np.inf,2.] if which=='gradient' else [1.,2.])
    assert c.state_dict()==before


def test_zero_gradients_and_corrupt_restart_rejected():
    c=WindowBalancer(['a','b'],BalanceConfig(interval=1))
    c.observe([1.,2.],[0.,0.])
    np.testing.assert_array_equal(c.weights,1)
    bad=c.state_dict();bad['last_probe']=100
    with pytest.raises(ValueError):WindowBalancer.from_state(bad)


def test_calibration_floor_and_fixed_coefficients():
    cal=calibrate([[0.,1.,1000.],[0.,3.,1000.]],[0.,.5,.5])
    assert min(cal['scales'])>0 and np.isfinite(cal['coefficients']).all()
    assert cal['coefficients'][0]==0
    # Floor = 1% of median([2,1000]) = 5.01; the smaller group is deliberately
    # NOT scaled all the way up to its requested priority.
    np.testing.assert_allclose(cal['floor'],5.01)
    np.testing.assert_allclose(np.asarray(cal['coefficients'])*cal['mean_groups'],[0,.5*2/5.01,.5])


@pytest.mark.parametrize('kw',[{'window':0},{'interval':0},{'alpha':float('nan')},{'minimum':2.},{'max_change':1.}])
def test_bad_configuration(kw):
    with pytest.raises(ValueError):BalanceConfig(**kw)


def test_projection_respects_all_constraints_for_extreme_scales():
    x=bounded_mean_one([1e-100,1,1e100],np.array([.8,.8,.8]),np.array([1.2,1.2,1.2]))
    np.testing.assert_allclose(x.mean(),1,atol=1e-12)
    assert np.all(x>=.8-1e-12) and np.all(x<=1.2+1e-12)


def checkpoint_fixture():
    import optax
    params={'x':jnp.ones(2)}
    opt=optax.adam(1e-3).init(params)
    state=dict(schema='adaptive_resume_v1',identity='test',update=0,params=params,
        optimizer=opt,rng=jax.random.PRNGKey(22),controller=WindowBalancer(['a','b']).state_dict(),
        metrics=[],validations=[dict(update=0,eligible=True,selection_score=1.)],
        calibration=dict(coefficients=[1.,1.]),
        best=dict(update=0,selection_score=1.,params=params))
    return state,params,opt


def test_initial_candidate_and_checkpoint_view_recovery(tmp_path):
    from experimental.adaptive_loss.checkpoint import validate,commit,materialize
    state,params,opt=checkpoint_fixture()
    validate(state,'test',params,opt,6)
    commit(tmp_path,state)
    (tmp_path/'metrics.jsonl').write_text('torn garbage')
    (tmp_path/'best.pkl').write_bytes(b'invalid view')
    materialize(tmp_path,state)
    assert (tmp_path/'metrics.jsonl').read_text()==''
    import pickle
    assert pickle.loads((tmp_path/'best.pkl').read_bytes())['update']==0


@pytest.mark.parametrize('mutation',['identity','controller','optimizer','best'])
def test_checkpoint_rejects_inconsistent_resume(mutation):
    from experimental.adaptive_loss.checkpoint import validate
    state,params,opt=checkpoint_fixture()
    if mutation=='identity':state['identity']='wrong'
    elif mutation=='controller':state['controller']['update']=1
    elif mutation=='optimizer':state['update']=1
    elif mutation=='best':state['best']['selection_score']=.5
    with pytest.raises(ValueError):validate(state,'test',params,opt,6)
