"""Fixed A/B/N/R/Y/H/S/G feature order; all field arrays are [v,p,lon,lat]."""
import numpy as np
from graphcast import data_utils, solar_radiation
from .config import FIELDS, LEVELS, FORCING_FIELDS
from .artifacts import digest


def channel_names(arm='main'):
    names = [f'{block}/{v}/{p}' for block in (('A','B') if arm=='C4' else ('A','B','N')) for v in FIELDS for p in LEVELS]
    names += [f'R/{t}' for t in ('older','newer','target')]
    names += [f'{block}/{t}/{phase}' for block in ('Y','H') for t in ('older','newer','target') for phase in ('sin','cos')]
    names += ['S/sea_surface_temperature','S/sea_ice_cover','G/surface_geopotential','G/land_sea_mask']
    if arm=='C3':
        names.append('lead_hours/240')
    return names


def layout_identity(arm='main'):
    return digest(channel_names(arm))


def weather_channels(weather):
    return np.asarray(weather).reshape(259,*weather.shape[-2:]).transpose(1,2,0)


def grid_nodes(features):
    import jax.numpy as jnp
    return jnp.transpose(features,(1,0,2)).reshape(-1,1,features.shape[-1])


def from_grid_nodes(nodes, nlon, nlat):
    return nodes[:,0].reshape(nlat,nlon,-1).transpose(1,0,2)


def static_fields(model):
    aux = model._structure.specs.aux_features['xarray_dataset']
    values = np.stack([np.asarray(aux[k].transpose('longitude','latitude'))
                      for k in ('geopotential_at_surface','land_sea_mask')],axis=-1)
    if values.shape != (128,64,2) or not np.isfinite(values).all():
        raise ValueError('Checkpoint static fields are not on the expected data grid')
    return values.astype(np.float32)


def time_channels(times, latitude, longitude):
    times = np.asarray(times,dtype='datetime64[s]')
    lat,lon = np.rad2deg(latitude),np.rad2deg(longitude)
    solar = np.asarray(solar_radiation.get_toa_incident_solar_radiation(
        times,lat,lon,use_jit=True)).transpose(2,1,0)
    seconds = times.astype(np.int64)
    year = data_utils.featurize_progress('year',('time',),data_utils.get_year_progress(seconds))
    day = data_utils.featurize_progress('day',('time','lon'),data_utils.get_day_progress(seconds,lon))
    shape = (len(lon),len(lat))
    y = np.stack([np.broadcast_to(np.asarray(year['year_'+phase])[t],shape)
                  for t in range(len(times)) for phase in ('sin','cos')],axis=-1)
    h = np.stack([np.broadcast_to(np.asarray(day['day_'+phase])[t,:,None],shape)
                  for t in range(len(times)) for phase in ('sin','cos')],axis=-1)
    return np.concatenate((solar,y,h),axis=-1).astype(np.float32)


def ancillary(times, forcing, static, latitude, longitude):
    surfaces = np.stack([np.asarray(forcing[k]).reshape(static.shape[:2]) for k in FORCING_FIELDS],axis=-1)
    return np.concatenate((time_channels(times,latitude,longitude),surfaces,static),axis=-1)


def assemble(older,newer,baseline,extra,stats,config,lead=6):
    weather = [older,newer] if config.arm=='C4' else [older,newer,baseline]
    blocks = [weather_channels((x-stats['mean'][...,None,None])/stats['std'][...,None,None]) for x in weather]
    blocks.append((extra-stats['extra_mean'])/stats['extra_std'])
    if config.arm=='C3':
        blocks.append(np.full((*older.shape[-2:],1),lead/240,dtype=np.float32))
    result = np.concatenate(blocks,axis=-1).astype(np.float32)
    if result.shape[-1] != config.channels or not np.isfinite(result).all():
        raise ValueError('Invalid feature layout or nonfinite normalized features')
    return result
