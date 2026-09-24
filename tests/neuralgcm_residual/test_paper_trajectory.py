from types import SimpleNamespace
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

from src.models.neuralgcm_residual.config import FIELDS
from src.models.neuralgcm_residual.native_state import NATIVE_FIELDS
from src.models.neuralgcm_residual.paper_loss import PaperLoss
from src.models.neuralgcm_residual.paper_statistics import PopulationMoments
from src.models.neuralgcm_residual.trajectory_training import TrajectoryTrainer


def loss_fixture(steps=2):
    grid=SimpleNamespace(modal_shape=(1,1),mask=np.ones((1,1)),radius=1.)
    model=SimpleNamespace(data_coords=SimpleNamespace(horizontal=grid),
                          model_coords=SimpleNamespace(horizontal=grid))
    names={'data':FIELDS,'model':tuple(k for k,_ in NATIVE_FIELDS)}
    stats={'split':'train','lag_hours':24,'scales':{s:{k:[1.] for k in ns} for s,ns in names.items()}}
    loss=PaperLoss(model,stats,steps)
    zero={s:{k:jnp.zeros((2,steps,1,1,1)) for k in ns} for s,ns in names.items()}
    return loss,zero


def test_batch_bias_averages_before_square():
    loss,p=loss_fixture()
    _,t=loss_fixture()
    p['data']['temperature']=jnp.array([0.,2.]).reshape(2,1,1,1,1)*jnp.ones((1,2,1,1,1))
    t['data']['temperature']=jnp.ones((2,2,1,1,1))
    value,details=loss.terms(p,t)
    np.testing.assert_allclose(details['terms']['bias'],0,atol=1e-8)
    expected=20*np.mean(1/(1+np.array([6,12])/24))/(4*np.pi)
    np.testing.assert_allclose(details['terms']['data'],expected,rtol=2e-6)
    assert float(value)>float(details['terms']['data'])


def test_cloud_amplitude_is_squared_and_zero_spectrum_gradient_finite():
    loss,p=loss_fixture(1)
    _,target=loss_fixture(1)
    p['data']['temperature']=jnp.ones((2,1,1,1,1))
    t_score=float(loss(p,target))
    p['data']['temperature']=jnp.zeros((2,1,1,1,1))
    p['data']['specific_cloud_liquid_water_content']=jnp.ones((2,1,1,1,1))
    np.testing.assert_allclose(float(loss(p,target))/t_score,.05**2,rtol=2e-6)
    g=jax.grad(loss)(target,target)
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(g))


def test_population_pooling_includes_mean_shifts():
    arrays=[np.arange(24).reshape(2,3,4),np.arange(24).reshape(2,3,4)+20]
    for per_level in (False,True):
        m=PopulationMoments(per_level)
        for x in arrays:m.add(x)
        axes=(0,2,3) if per_level else (0,1,2,3)
        np.testing.assert_allclose(m.std(),np.atleast_1d(np.stack(arrays).std(axis=axes)))


class State(NamedTuple):
    temperature_variation:object
    vorticity:object
    divergence:object
    log_surface_pressure:object
    tracers:dict


def state(x):
    return State(x,jnp.float32(0),jnp.float32(0),jnp.float32(0),
                 {k:jnp.float32(0) for k in FIELDS if k.startswith('specific_')})


def test_memory_only_reverse_matches_independent_batch_trajectory_gradient():
    class Branch:
        def apply(self,p,h,key,x,known):
            h={'h':p['a']*h['h']+p['b']*x}
            return p['c']*h['h'],h
    class Backbone:
        model=SimpleNamespace(data_coords=SimpleNamespace(horizontal=SimpleNamespace(to_modal=lambda x:x)),
                              datetime64_to_sim_time=lambda t:np.float32(0))
        def encode(self,x,f):return state(x['temperature'])
        def advance_6h(self,s,f):return s._replace(temperature_variation=s.temperature_variation*1.02+.1)
        def decode(self,s,f):return {k:s.temperature_variation**2 for k in FIELDS}
    class Store:
        def inputs_and_forcing(self,model,t):return {'temperature':jnp.float32(.2+int(t)*.1)},jnp.float32(1)
        def frame(self,t):return {k:jnp.float32(.3) for k in FIELDS}
    class Loss:
        steps=2
        def terms(self,p,t):
            e=p['data']['temperature']-t['data']['temperature']
            value=jnp.mean(e**2)+.3*jnp.mean(e)**2
            return value,{'value':value}
    b=Backbone()
    r=SimpleNamespace(backbone=b,branch=Branch(),store=Store(),memory={'h':jnp.float32(0)},
        adapter=SimpleNamespace(features=lambda s:s.temperature_variation,
            apply_increment=lambda s,d:s._replace(temperature_variation=s.temperature_variation+d)),
        normalization=SimpleNamespace(inputs=lambda x:x,increment=lambda x:x),known=lambda s,f:jnp.float32(0))
    trainer=TrajectoryTrainer(r,Loss())
    p={k:jnp.float32(v) for k,v in dict(a=.8,b=.2,c=.1).items()}
    result=trainer.batch(p,jax.random.PRNGKey(22),['0','1'])
    def reference(p):
        errors=[]
        for origin in (0,1):
            physical=jnp.float32(.2+origin*.1);h={'h':jnp.float32(0)}
            for _ in range(2):
                physical=jax.lax.stop_gradient(physical)
                d,h=r.branch.apply(p,h,None,physical,0)
                physical=physical*1.02+.1+d
                errors.append(physical**2-.3)
        e=jnp.stack(errors)
        return jnp.mean(e**2)+.3*jnp.mean(e)**2
    expected=jax.grad(reference)(p)
    np.testing.assert_allclose(result[0],reference(p),rtol=1e-6)
    for name in p:
        np.testing.assert_allclose(result[3][name],expected[name],rtol=2e-5,atol=1e-7)
    assert abs(float(result[3]['a']))>1e-6
