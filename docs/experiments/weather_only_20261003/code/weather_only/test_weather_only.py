"""CPU numerical tests of strict output masking, independent of production data."""
import unittest
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from src.models.neuralgcm_residual.config import FIELDS
from src.models.neuralgcm_residual.native_state import NATIVE_FIELDS
from experimental.adaptive_loss.objective import GroupedLoss
from experimental.weather_only.objective import WeatherOnlyLoss,calibrate

class MaskTests(unittest.TestCase):
    def setUp(self):
        self.pressure=np.array([1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,350,400,450,500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000])
        grid=SimpleNamespace(radius=1.,mask=np.ones((4,5)),modal_shape=(4,5))
        coords=SimpleNamespace(horizontal=grid,vertical=SimpleNamespace(centers=self.pressure))
        model=SimpleNamespace(data_coords=coords,model_coords=coords)
        stats=dict(split='train',lag_hours=24,scales={'data':{f:[2.] for f in FIELDS},'model':{f:[2.] for f,_ in NATIVE_FIELDS}})
        self.new=WeatherOnlyLoss(model,stats,2);self.old=GroupedLoss(model,stats,2)
        rng=np.random.default_rng(12)
        def tree():return {'data':{f:jnp.asarray(rng.normal(size=(2,2,37,4,5)),jnp.float32) for f in FIELDS},'model':{f:jnp.asarray(rng.normal(size=(2,2,32,4,5)),jnp.float32) for f,_ in NATIVE_FIELDS}}
        self.pred,self.truth=tree(),tree()
    def test_selected_component_equivalence(self):
        old=self.old.components(self.pred,self.truth)
        indices=[i for i,n in enumerate(self.old.names) if n.endswith('/weather')]
        np.testing.assert_allclose(self.new.components(self.pred,self.truth),np.asarray(old)[indices]*37/29,rtol=3e-6)
    def test_exact_excluded_cotangents(self):
        grad=jax.grad(lambda x:self.new.terms(x,self.truth)[0])(self.pred)
        for x in grad['data'].values():self.assertEqual(np.count_nonzero(np.asarray(x)[:,:,:8]),0)
        for x in grad['model'].values():self.assertEqual(np.count_nonzero(x),0)
        self.assertGreater(sum(float(jnp.sum(x[:,:,8:]**2)) for x in grad['data'].values()),0.)
    def test_excluded_values_cannot_change_calibration_or_loss(self):
        alt=jax.tree_util.tree_map(lambda x:x,self.pred)
        alt['data']={n:x.at[:,:,:8].set(1e10) for n,x in alt['data'].items()}
        alt['model']=jax.tree_util.tree_map(lambda x:x*1e10,alt['model'])
        a,b=self.new.components(self.pred,self.truth),self.new.components(alt,self.truth)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(calibrate([a],self.new.priors),calibrate([b],self.new.priors))
    def test_window_weights_have_only_selected_groups(self):
        from experimental.adaptive_loss.controller import WindowBalancer,BalanceConfig
        c=WindowBalancer(self.new.names,BalanceConfig(window=4,interval=2))
        c.observe(np.ones(7));c.observe(np.ones(7),np.arange(1,8))
        self.assertEqual(len(c.weights),7);self.assertTrue(all(n.endswith('/weather') for n in c.names))
        self.assertAlmostEqual(c.weights.mean(),1.);self.assertGreater(np.ptp(c.weights),0.)
        weights=jnp.asarray(c.weights)
        grad=jax.grad(lambda x:self.new.weighted(x,self.truth,weights)[0])(self.pred)
        for x in grad['data'].values():self.assertEqual(np.count_nonzero(np.asarray(x)[:,:,:8]),0)
    def test_dynamic_coefficients_jit(self):
        fn=jax.jit(lambda w:self.new.weighted(self.pred,self.truth,w)[0])
        np.testing.assert_allclose(fn(jnp.ones(7)*2),2*fn(jnp.ones(7)),rtol=1e-6)

if __name__=='__main__':unittest.main()
