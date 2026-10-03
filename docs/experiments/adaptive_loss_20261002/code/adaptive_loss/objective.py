"""Exact field/band decomposition of the existing five-term objective.

No changing weights are captured in a JIT closure. They are explicit inputs.
Data groups split <=30 hPa and >30 hPa, preserving the full-level denominator.
Native log pressure remains in fixed evaluation, but has zero adaptive prior
because this experiment's physical stop-gradient makes it untrainable.
"""
import jax
import jax.numpy as jnp
import numpy as np
from neuralgcm.legacy.model_utils import safe_sqrt

from src.models.neuralgcm_residual.paper_loss import PaperLoss, amplitude


class GroupedLoss(PaperLoss):
    def __init__(self, model, statistics, steps, *, beta=1e-5):
        super().__init__(model, statistics, steps)
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError('Base upper-pressure coefficient must be positive')
        self.beta = beta
        pressure = np.asarray(model.data_coords.vertical.centers, np.float64)
        if pressure.ndim != 1 or not np.isfinite(pressure).all() or np.any(pressure <= 0):
            raise ValueError('Invalid pressure coordinates')
        self.bands = {'upper': pressure <= 30., 'weather': pressure > 30.}
        if not all(x.any() for x in self.bands.values()):
            raise ValueError('Both pressure bands must contain levels')
        self.names = tuple(f'data/{name}/{band}' for name in self.fields['data'] for band in self.bands)
        self.names += tuple('model/'+name for name in self.fields['model'])
        self.excluded = ('model/log_surface_pressure',)
        # Explicit scientific priorities, separate from numerical normalization.
        # 90% data / 10% trainable native; upper data 5% / weather data 95%.
        # Each cloud field receives 0.05 of a main data variable's priority.
        raw = np.array([.05 if n.startswith('specific_cloud_') else 1. for n in self.fields['data']])
        field_prior = dict(zip(self.fields['data'], raw/raw.sum(), strict=True))
        prior = [.9*field_prior[n]*(.05 if band == 'upper' else .95)
                 for n in self.fields['data'] for band in self.bands]
        prior += [0. if n == 'log_surface_pressure' else .1/(len(self.fields['model'])-1)
                  for n in self.fields['model']]
        self.priors = np.asarray(prior, np.float32)

    def components(self, prediction, target):
        """Return disjoint group contributions; their sum equals fixed beta loss."""
        groups = []
        for space, names in self.fields.items():
            grid = self.grids[space]
            area = jnp.float32(4*np.pi*grid.radius**2)
            mask = jnp.asarray(grid.mask)
            for name in names:
                scale = amplitude(name)/self.scales[space][name]
                p, t = prediction[space][name]*scale*mask, target[space][name]*scale*mask
                err = (p-t)*self.time[None,:,None,None,None]*self.filters[space]
                acc = jnp.mean(jnp.sum(err**2, axis=(-2,-1))/area, axis=(0,1))
                ps, ts = safe_sqrt(jnp.sum(p**2,axis=-2)), safe_sqrt(jnp.sum(t**2,axis=-2))
                se = (ps-ts)[...,:43]*self.spectral_time[None,:,None,None]
                spec = jnp.mean(jnp.sum(se**2,axis=-1)/area, axis=(0,1))
                per_level = (20. if space == 'data' else 1.)*acc+.1*spec
                if space == 'data':
                    e = (jnp.abs(p)-jnp.abs(t))*self.time[None,:,None,None,None]
                    bias = jnp.sum(jnp.mean(e,axis=(0,1))**2,axis=(-2,-1))/area
                    per_level = per_level+2*bias
                    for band, selection in self.bands.items():
                        groups.append(jnp.mean(per_level*jnp.asarray(selection))*(self.beta if band == 'upper' else 1.))
                else:
                    groups.append(jnp.mean(per_level))
        return jnp.stack(groups)

    def weighted(self, prediction, target, coefficients):
        groups = self.components(prediction,target)
        # The controller is external; weights are not trainable model parameters.
        value = jnp.dot(groups, jax.lax.stop_gradient(coefficients))
        return value, {'groups': groups, 'fixed_beta_loss': jnp.sum(groups)}


def calibrate(group_samples, priors):
    """Fixed train-only scales. Relative floor avoids dividing by near-zero MSE."""
    samples, priors = np.asarray(group_samples,np.float64), np.asarray(priors,np.float64)
    if samples.ndim != 2 or samples.shape[1:] != priors.shape or not np.isfinite(samples).all() or np.any(samples < 0):
        raise ValueError('Invalid calibration samples')
    mean = samples.mean(axis=0)
    positive = mean[mean > 0]
    floor = max(1e-8, .01*float(np.median(positive))) if positive.size else 1e-8
    scales = np.maximum(mean, floor)
    coefficients = priors/scales
    return dict(scales=scales.tolist(), coefficients=coefficients.tolist(), floor=floor,
                mean_groups=mean.tolist(), samples=int(len(samples)))
