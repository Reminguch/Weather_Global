"""Five-term deterministic NeuralGCM objective on complete modal trajectories.

This is a versioned reconstruction. Statistics, filtering, field/level choices
and reductions are explicit artifacts, not inferred from an inference checkpoint.
Array axes are [batch, lead, level, zonal_wavenumber, total_wavenumber].
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from neuralgcm.legacy.model_utils import safe_sqrt

from .config import FIELDS
from .native_state import NATIVE_FIELDS, field_value

NAME = 'neuralgcm_five_term_shortlead_reconstruction_v1'
COEFFICIENTS = dict(data=20., model=1., data_spectrum=.1, model_spectrum=.1, bias=2.)


def amplitude(name):
    if name == 'geopotential':
        return 2.
    if name == 'specific_humidity':
        return .66
    if name.startswith('specific_cloud_'):
        return .05
    if name == 'log_surface_pressure':
        return 5.
    return 1.


def modal_prediction(backbone, state, forcing):
    decoded = backbone.decode(state, forcing)
    grid = backbone.model.data_coords.horizontal
    return {'data': {k: grid.to_modal(decoded[k]) for k in FIELDS},
            'model': {k: field_value(state, k) for k, _ in NATIVE_FIELDS}}


class PaperLoss:
    def __init__(self, model, statistics, steps, *, filter_half_wavenumber=120.):
        if steps not in (1, 2):
            raise ValueError('Short-lead reconstruction is defined only for K=1 or K=2')
        if statistics['split'] != 'train' or statistics['lag_hours'] != 24:
            raise ValueError('Paper loss requires training-only 24 h difference statistics')
        self.steps = steps
        self.grids = {'data': model.data_coords.horizontal, 'model': model.model_coords.horizontal}
        self.fields = {'data': FIELDS, 'model': tuple(k for k, _ in NATIVE_FIELDS)}
        self.scales = {}
        for space, names in self.fields.items():
            self.scales[space] = {}
            for name in names:
                x = np.asarray(statistics['scales'][space][name], np.float32)
                if x.ndim != 1 or not np.isfinite(x).all() or np.any(x <= 0):
                    raise ValueError(f'Invalid {space}/{name} loss scale')
                self.scales[space][name] = jnp.asarray(x)[None, None, :, None, None]
        if not np.isfinite(filter_half_wavenumber) or filter_half_wavenumber <= 0:
            raise ValueError('Filter half-amplitude wavenumber must be finite and positive')
        self.filters = {k: jnp.exp(-jnp.log(2.) *
            (jnp.arange(g.modal_shape[-1], dtype=jnp.float32) / filter_half_wavenumber)**24)
                        for k, g in self.grids.items()}
        lead = jnp.arange(1, steps+1, dtype=jnp.float32) * 6
        self.time = (1 + lead/24)**-.5
        self.spectral_time = (1 + (lead/40)**4)**-.5
        self.metadata = dict(name=NAME, coefficients=COEFFICIENTS,
            field_reduction='sum', level_reduction='uniform mean', time_reduction='mean',
            scored_hours=list(range(6, 6*steps+1, 6)), include_initial_time=False,
            spectrum_max_wavenumber=42, spectrum_reduction='sum_l / sphere_area; mean_level_batch_time',
            bias='squared batch-and-time mean modal amplitude difference, data representation',
            filter=dict(order=12, half_amplitude_wavenumber=filter_half_wavenumber,
                        scope='explicit short-lead approximation; original bindings unavailable'))

    def terms(self, prediction, target):
        terms = {k: jnp.float32(0) for k in COEFFICIENTS}
        fields = {}
        for space, names in self.fields.items():
            grid = self.grids[space]
            mask = jnp.asarray(grid.mask)
            area = jnp.float32(4*np.pi*grid.radius**2)
            for name in names:
                scale = amplitude(name) / self.scales[space][name]
                p, t = prediction[space][name]*scale, target[space][name]*scale
                # Mask padding before all reductions, including spectra/bias.
                p, t = p*mask, t*mask
                err = (p-t)*self.time[None, :, None, None, None]*self.filters[space]
                acc = jnp.mean(jnp.sum(err**2, axis=(-2,-1))/area)
                ps = safe_sqrt(jnp.sum(p**2, axis=-2))
                ts = safe_sqrt(jnp.sum(t**2, axis=-2))
                se = (ps-ts)[..., :43]*self.spectral_time[None, :, None, None]
                spec = jnp.mean(jnp.sum(se**2, axis=-1)/area)
                terms[space] += acc
                terms[space+'_spectrum'] += spec
                fields[space+'/'+name] = COEFFICIENTS[space]*acc + .1*spec
                if space == 'data':
                    # Reference BatchMeanSquaredBias defaults to abs(modal),
                    # then averages batch and time BEFORE the squared norm.
                    bias_err = (jnp.abs(p)-jnp.abs(t))*self.time[None, :, None, None, None]
                    bias = jnp.mean(jnp.sum(jnp.mean(bias_err, axis=(0,1))**2, axis=(-2,-1))/area)
                    terms['bias'] += bias
                    fields[space+'/'+name] += 2*bias
        weighted = {k: COEFFICIENTS[k]*v for k,v in terms.items()}
        return sum(weighted.values()), {'terms':weighted, 'fields':fields}

    def __call__(self, prediction, target):
        return self.terms(prediction, target)[0]
