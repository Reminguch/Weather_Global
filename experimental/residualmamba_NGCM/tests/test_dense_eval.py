import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from residualmamba_ngcm.config import Config
from residualmamba_ngcm.train import validate,run
from residualmamba_ngcm.artifacts import load_checkpoint
from dense_eval import predictor,evaluate_segment

LAT=np.arcsin(np.polynomial.legendre.leggauss(4)[0])


def tiny_model():
    def forward(x):
        p=hk.get_parameter('p',(),init=hk.initializers.Constant(.2))
        h=hk.get_state('h',(),init=hk.initializers.Constant(0))
        new=jnp.tanh(h*p+x);hk.set_state('h',new)
        return jnp.broadcast_to(new*p,(7,29,1,4))
    return hk.transform_with_state(forward)


def test_dense_evaluator_matches_qualified_validation_and_reset():
    model=tiny_model();params,memory=model.init(jax.random.PRNGKey(22),.1)
    batch={'features':np.linspace(.1,.5,24,dtype=np.float32),'target':np.ones((24,7,29,1,4),np.float32)*.3}
    load=lambda _:batch;segment=np.arange(96)
    expected=validate(model,params,memory,[segment],load,LAT,Config())
    loss,baseline=evaluate_segment(predictor(model,LAT,22),params,memory,segment,load,LAT,22)
    np.testing.assert_allclose(loss.sum(axis=(1,2)),expected['per_step'],rtol=2e-6,atol=1e-8)
    np.testing.assert_allclose(baseline.sum(axis=(1,2)),expected['baseline_per_step'],rtol=2e-6,atol=1e-8)
    changed=jax.tree_util.tree_map(lambda x:x+100,memory)
    other,_=evaluate_segment(predictor(model,LAT,22),params,changed,segment,load,LAT,22)
    np.testing.assert_array_equal(loss,other)


def test_reentering_trainer_preserves_every_state(tmp_path):
    model=tiny_model();batch={'features':np.linspace(.1,.5,24,dtype=np.float32),'target':np.ones((24,7,29,1,4),np.float32)*.1}
    segments=(np.arange(96),np.arange(96,192));load=lambda _:batch;cfg=Config()
    run(model,cfg,segments,(),load,LAT,tmp_path/'full','dense-test',steps=5,defer_validation=True)
    resume=None
    for expected in (2,4,5):
        result=run(model,cfg,segments,(),load,LAT,tmp_path/'split','dense-test',steps=5,resume=resume,
            stop_after_update=lambda r:r['step']%2==0,defer_validation=True)
        assert result['step']==expected
        resume=tmp_path/f'split/step_{expected:05d}.pkl'
    a=load_checkpoint(tmp_path/'full/step_00005.pkl');b=load_checkpoint(resume)
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):np.testing.assert_array_equal(x,y)
