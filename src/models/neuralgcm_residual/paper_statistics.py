"""Train-only 24 h statistics, including linearly interpolated native fields."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from dinosaur import spherical_harmonic, vertical_interpolation

from .config import FIELDS
from .data import split_for, stamp
from .io import digest


class PopulationMoments:
    def __init__(self, per_level):
        self.per_level, self.count = per_level, 0
        self.mean, self.m2 = 0., 0.

    def add(self, x):
        x = np.asarray(x,np.float64)
        if x.ndim != 3 or not np.isfinite(x).all():
            raise ValueError('Expected finite [level, longitude, latitude] samples')
        axes = (1,2) if self.per_level else (0,1,2)
        mean, var = x.mean(axis=axes), x.var(axis=axes)
        self.count += 1
        shift = mean-self.mean
        self.mean = self.mean+shift/self.count
        self.m2 = self.m2+var+shift*(mean-self.mean)

    def std(self):
        x = np.atleast_1d(np.sqrt(self.m2/self.count))
        if not np.isfinite(x).all() or np.any(x <= 0):
            raise ValueError('Degenerate scale; no arbitrary loss floor is applied')
        return x.tolist()


def linear_native(model):
    """Physical pressure-to-sigma interpolation without learned encoding.

Reference surface geopotential is taken from the checkpoint auxiliary dataset.
This reconstruction does not include its learned orography perturbation.
"""
    data, coords = model.data_coords, model.model_coords
    if data.horizontal.nodal_shape != coords.horizontal.nodal_shape:
        raise ValueError('Initial statistics implementation requires matching horizontal grids')
    aux = model._structure.specs.aux_features['xarray_dataset']
    orography = jnp.asarray(aux.geopotential_at_surface.transpose('longitude','latitude').values)/9.80665
    interpolation = vertical_interpolation.vectorize_vertical_interpolation(
        vertical_interpolation.vertical_interpolation)

    @jax.jit
    def convert(frame):
        ps = vertical_interpolation.get_surface_pressure(data.vertical,
            frame['geopotential'],orography=orography,gravity_acceleration=9.80665)
        fields = vertical_interpolation.interp_pressure_to_sigma(
            {k:frame[k] for k in FIELDS},data.vertical,coords.vertical,ps,interpolation)
        u = model.to_nondim_units(fields['u_component_of_wind'],'m/s')
        v = model.to_nondim_units(fields['v_component_of_wind'],'m/s')
        vor,div = spherical_harmonic.uv_nodal_to_vor_div_modal(coords.horizontal,u,v)
        return dict(vorticity=coords.horizontal.to_nodal(vor),
                    divergence=coords.horizontal.to_nodal(div),
                    temperature_variation=fields['temperature'],
                    log_surface_pressure=jnp.log(ps),
                    **{k:fields[k] for k in FIELDS if k.startswith('specific_')})
    return convert


def fit_statistics(model, store, samples=60, *, progress=None):
    available = set(store.times)
    candidates = [t for t in store.times if split_for(t)=='train' and
                  t+np.timedelta64(24,'h') in available and
                  split_for(t+np.timedelta64(24,'h'))=='train']
    if not 2 <= samples <= len(candidates):
        raise ValueError('Invalid calibration sample count')
    origins = [candidates[i] for i in np.linspace(0,len(candidates)-1,samples,dtype=int)]
    moments = {'data':{},'model':{}}
    convert = linear_native(model)
    hashes = {}
    for i,origin in enumerate(origins):
        frames = [store.frame(t) for t in (origin,origin+np.timedelta64(24,'h'))]
        native = [jax.device_get(convert({k:x[k] for k in FIELDS})) for x in frames]
        for space, pair in [('data',frames),('model',native)]:
            names = FIELDS if space=='data' else native[0].keys()
            for name in names:
                if name not in moments[space]:
                    moments[space][name] = PopulationMoments(name=='specific_humidity')
                moments[space][name].add(pair[1][name]-pair[0][name])
        for t in (origin,origin+np.timedelta64(24,'h')):
            hashes[stamp(t)] = store.records[stamp(t)]['sha256']
        if progress:
            progress({'statistics_samples':i+1,'total':samples})
    return dict(version='ngcm_24h_data_linear_native_v1',split='train',lag_hours=24,
        samples=samples,dataset_id=store.identity,origins=list(map(stamp,origins)),
        source_frame_sha256=hashes,source_frames_identity=digest(hashes),
        spatial_statistics='uniform gridpoint population moments',
        pooling='samples, all levels and gridpoints; specific humidity separately per level',
        scale_floors=None,native_conversion='linear pressure-to-sigma; auxiliary orography without learned perturbation',
        scales={space:{k:v.std() for k,v in values.items()} for space,values in moments.items()})
