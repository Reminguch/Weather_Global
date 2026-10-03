"""Seven data-only objectives with <=30 hPa structurally absent.

Native-state losses are absent because sigma levels do not map to a fixed
pressure mask. The solver, decoder and all 37 evaluation levels are unchanged.
"""
import jax
import jax.numpy as jnp
import numpy as np
from neuralgcm.legacy.model_utils import safe_sqrt
from src.models.neuralgcm_residual.paper_loss import PaperLoss, amplitude
from experimental.adaptive_loss.objective import calibrate

class WeatherOnlyLoss(PaperLoss):
    def __init__(self, model, statistics, steps):
        super().__init__(model, statistics, steps)
        pressure=np.asarray(model.data_coords.vertical.centers,float)
        if pressure.ndim!=1 or not np.isfinite(pressure).all() or np.any(pressure<=0):
            raise ValueError('Invalid pressure coordinates')
        self.indices=np.flatnonzero(pressure>30.)
        if not 0<len(self.indices)<len(pressure):raise ValueError('Both pressure bands required')
        self.names=tuple('data/'+n+'/weather' for n in self.fields['data'])
        self.excluded=()
        raw=np.array([.05 if n.startswith('specific_cloud_') else 1. for n in self.fields['data']])
        self.priors=(raw/raw.sum()).astype(np.float32)
        self.metadata.update(name='weather_only_data_v1',pressure_mask='>30 hPa',native_loss_weight=0.,upper_loss_weight=0.)

    def components(self, prediction, target):
        grid=self.grids['data'];area=jnp.float32(4*np.pi*grid.radius**2);mask=jnp.asarray(grid.mask)
        groups=[]
        for name in self.fields['data']:
            # Slice BEFORE any arithmetic; excluded cells cannot enter normalization,
            # spectral/bias objectives or cotangents. No multiplying a bad value by zero.
            scale=self.scales['data'][name]
            if scale.shape[2]!=1:scale=jnp.take(scale,self.indices,axis=2)
            scale=amplitude(name)/scale
            p=jnp.take(prediction['data'][name],self.indices,axis=2)*scale*mask
            t=jnp.take(target['data'][name],self.indices,axis=2)*scale*mask
            err=(p-t)*self.time[None,:,None,None,None]*self.filters['data']
            acc=jnp.mean(jnp.sum(err**2,axis=(-2,-1))/area)
            ps,ts=safe_sqrt(jnp.sum(p**2,axis=-2)),safe_sqrt(jnp.sum(t**2,axis=-2))
            se=(ps-ts)[...,:43]*self.spectral_time[None,:,None,None]
            spec=jnp.mean(jnp.sum(se**2,axis=-1)/area)
            e=(jnp.abs(p)-jnp.abs(t))*self.time[None,:,None,None,None]
            bias=jnp.mean(jnp.sum(jnp.mean(e,axis=(0,1))**2,axis=(-2,-1))/area)
            groups.append(20*acc+.1*spec+2*bias)
        return jnp.stack(groups)

    def terms(self,prediction,target):
        groups=self.components(prediction,target)
        return jnp.sum(groups),dict(groups=groups)

    def weighted(self,prediction,target,coefficients):
        groups=self.components(prediction,target)
        return jnp.dot(groups,jax.lax.stop_gradient(coefficients)),dict(groups=groups)
