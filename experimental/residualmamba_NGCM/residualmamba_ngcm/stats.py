"""Streaming population moments on training truth, including adjacent 6h differences."""
import numpy as np
from .config import FIELDS, LEVELS
from .timeline import stack_weather
from .features import ancillary, static_fields
from .artifacts import save_arrays, source_identity, versions
from src.models.neuralgcm_residual.data import STEP, split_for


class Moments:
    """Parallel Welford merge, last axis contains independent samples."""
    def __init__(self):
        self.n = 0
        self.mean = None
        self.m2 = None

    def add(self, values):
        x = np.asarray(values,dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite statistics input')
        n,mean = x.shape[-1],x.mean(axis=-1)
        m2 = np.sum((x-mean[...,None])**2,axis=-1)
        if self.n == 0:
            self.n,self.mean,self.m2 = n,mean,m2
        else:
            delta = mean-self.mean
            total = self.n+n
            self.m2 += m2+delta**2*(self.n*n/total)
            self.mean += delta*(n/total)
            self.n = total

    def result(self):
        if not self.n:
            raise ValueError('Empty statistics sample')
        return self.mean,np.sqrt(self.m2/self.n)


def scale_floors(scale):
    scale = np.asarray(scale,dtype=np.float64).copy()
    records = []
    for v,name in enumerate(FIELDS):
        if not np.isfinite(scale[v]).all():
            raise ValueError(f'Nonfinite scale: {name}')
        positive = scale[v][scale[v]>0]
        if not len(positive):
            raise ValueError(f'No positive scale: {name}')
        if v >= 5:
            floor = .001*np.median(positive)
            levels = np.flatnonzero(scale[v]<floor)
            records.append({'field':name,'floor':float(floor),'levels_hpa':[LEVELS[p] for p in levels]})
            scale[v] = np.maximum(scale[v],floor)
        elif np.any(scale[v]<=0):
            raise ValueError(f'Nonpositive scale without an approved floor: {name}')
    return scale.astype(np.float32),records


def compute(bb,store,config,path,preflight_hash):
    weather,diffs,extra = Moments(),Moments(),Moments()
    static = static_fields(bb.model)
    grid = bb.model.data_coords.horizontal
    times = [t for t in store.times if split_for(t)=='train'][::config.stats_stride]
    diff_times,extra_times = [],[]
    climatology_sum = None
    for index,t in enumerate(times):
        x = stack_weather(store.frame(t))
        weather.add(x.reshape(7,37,-1))
        climatology_sum = x.astype(np.float64) if climatology_sum is None else climatology_sum+x
        if t+STEP in store.times and split_for(t+STEP)=='train':
            delta = stack_weather(store.frame(t+STEP))-x
            diffs.add(delta.reshape(7,37,-1))
            diff_times.append([str(t),str(t+STEP)])
        # All three feature times and the lagged surface forcing stay in train.
        if all(split_for(t+k*STEP)=='train' and t+k*STEP in store.times for k in (-4,-1,0,1)):
            _,forcing = store.inputs_and_forcing(bb.model,t)
            e = ancillary([t-STEP,t,t+STEP],forcing,static,grid.latitudes,grid.longitudes)
            extra.add(e.transpose(2,0,1).reshape(19,-1))
            extra_times.append(str(t))
        if index%100==0:
            print(f'stats {index+1}/{len(times)} {t}',flush=True)
    mean,std = weather.result()
    _,diff_std = diffs.result()
    std, floors = scale_floors(std)
    diff_std,diff_floors = scale_floors(diff_std)
    extra_mean,extra_std = extra.result()
    if np.any(extra_std<=0) or not np.isfinite(extra_std).all():
        raise ValueError('Ancillary statistics must have finite positive standard deviations')
    arrays = {'mean':mean.astype(np.float32),'std':std,'diff_std':diff_std,
              'extra_mean':extra_mean.astype(np.float32),'extra_std':extra_std.astype(np.float32),
              'climatology':(climatology_sum/len(times)).astype(np.float32)}
    manifest = {'stage':'stats','store_identity':store.identity,'preflight_hash':preflight_hash,
        'source_identity':source_identity(),'libraries':versions(),'stride':config.stats_stride,
        'train_times':[str(t) for t in times],'diff_pairs':diff_times,'extra_times':extra_times,
        'std_floors':floors,'diff_std_floors':diff_floors,'population_std':True,'spatial_weighting':'uniform',
        'climatology':'fixed all-training-time mean at each grid point'}
    save_arrays(path,arrays,manifest)
    return manifest
