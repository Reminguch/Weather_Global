from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.models.neuralgcm_residual.kernels import DecodedTrainer
from src.models.neuralgcm_residual.finetune import episode_gradients


class Branch:
    def apply(self,p,h,key,x,known):
        h={'h':p['a']*h['h']+p['b']*x}
        return p['c']*h['h'],h


class Backbone:
    model=None
    def encode(self,x,f): return x
    def advance_6h(self,s,f): return s*1.01+.1
    def decode(self,s,f): return s**2
    def assert_time(self,*args): pass


class Store:
    def inputs_and_forcing(self,model,t): return jnp.float32(.2),jnp.float32(1)
    def frame(self,t): return jnp.float32(.3)


def fixture():
    b=Backbone()
    adapter=SimpleNamespace(features=lambda s:s,apply_increment=lambda s,d:s+d)
    norm=SimpleNamespace(inputs=lambda x:x,increment=lambda x:x)
    trainer=DecodedTrainer(Branch(),adapter,norm,b,lambda p,t:(p-t)**2,optax.adam(1e-3))
    params={'a':jnp.float32(.8),'b':jnp.float32(.2),'c':jnp.float32(.1)}
    return trainer,params,{'h':jnp.float32(.7)},jax.random.PRNGKey(22)


def assert_tree(a,b):
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b),strict=True):
        np.testing.assert_allclose(x,y,rtol=2e-5,atol=1e-7)


def test_decoded_gradient_nonzero_memory_and_reverse_tape():
    trainer,p,h,key=fixture()
    records=[tuple(map(jnp.float32,[i/10,.0,.5,1,.4])) for i in range(1,7)]
    loss,next_h,rng,g=trainer.cached_gradients(p,h,key,records)
    def reference(params):
        memory=h
        losses=[]
        for record in records:
            l,memory,_=trainer.forward(params,memory,key,*record)
            losses.append(l)
        return jnp.mean(jnp.stack(losses))
    assert_tree(g,jax.grad(reference)(p))
    assert abs(float(g['c']))>1e-4  # frozen decoder input derivative reaches residual
    assert abs(float(g['a']))>1e-5  # recurrent derivative survives
    opt=trainer.optimizer.init(p)
    updated,_,_=trainer.checked_update(p,opt,g)
    assert float(updated['c'])!=float(p['c'])


def test_chunked_forward_carry_equals_contiguous():
    trainer,p,h,key=fixture()
    records=[tuple(map(jnp.float32,[i/10,.0,.5,1,.4])) for i in range(1,9)]
    def execute(memory,records):
        values=[]
        for record in records:
            l,memory,_=trainer.forward(p,memory,key,*record)
            values.append(l)
        return values,memory
    full,final=execute(h,records)
    first,carry=execute(h,records[:4])
    second,last=execute(jax.tree_util.tree_map(jax.lax.stop_gradient,carry),records[4:])
    assert_tree(full,first+second)
    assert_tree(final,last)


def test_live20_uses_corrected_state_but_no_physical_gradient():
    trainer,p,h,key=fixture()
    result=episode_gradients(trainer,p,h,key,backbone=trainer.backbone,store=Store(),
                             known=lambda s,f:jnp.float32(0),origin='2020-07-01T00')
    assert result['valid_hours']==list(range(6,121,6))
    def reference(params,sg):
        physical=jnp.float32(.2)
        memory={'h':jnp.float32(0)}
        losses=[]
        for _ in range(20):
            if sg: physical=jax.lax.stop_gradient(physical)
            baseline=trainer.backbone.advance_6h(physical,1)
            delta,memory=trainer.branch.apply(params,memory,key,physical,0)
            physical=baseline+delta
            losses.append((physical**2-.3)**2)
        return jnp.mean(jnp.stack(losses))
    assert_tree(result['gradients'],jax.grad(lambda p:reference(p,True))(p))
    full=jax.grad(lambda p:reference(p,False))(p)
    assert not np.isclose(result['gradients']['b'],full['b'])
