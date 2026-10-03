"""Shared physical-metric conventions for live evaluation (no changing loss weights)."""
import hashlib,json,os,tempfile
from pathlib import Path
import numpy as np

FIELDS=('temperature','geopotential','u_component_of_wind','v_component_of_wind','specific_humidity','specific_cloud_ice_water_content','specific_cloud_liquid_water_content')
LABELS=('Temperature','Geopotential','Zonal wind (u)','Meridional wind (v)','Specific humidity','Cloud ice','Cloud liquid water')

def read(path):return json.loads(Path(path).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def write(path,x):atomic(path,(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
def improvement(current,baseline):
    current,baseline=np.asarray(current,float),np.asarray(baseline,float)
    floor=max(1e-12,1e-6*float(np.max(baseline)))
    valid=np.isfinite(current)&np.isfinite(baseline)&(baseline>floor)
    value=np.full(current.shape,np.nan);np.divide(100*(baseline-current),baseline,out=value,where=valid)
    return value

def counts(values,tolerance=.1):
    x=np.asarray(values,float)
    return dict(improved=int(np.sum(x>tolerance)),worse=int(np.sum(x<-tolerance)),
                unchanged=int(np.sum(np.isfinite(x)&(np.abs(x)<=tolerance))),undefined=int(np.sum(~np.isfinite(x))))

def reduce_fields(rmse,baseline,pressure,lead_index,band='all_levels',tolerance=.1):
    levels=np.asarray(pressure,float);mask=np.ones(len(levels),bool) if band=='all_levels' else levels>30
    summary={};cells=[]
    for field in FIELDS:
        current=np.asarray(rmse[field],float)[lead_index];reference=np.asarray(baseline[field],float)[lead_index]
        if current.shape!=levels.shape or reference.shape!=levels.shape:raise ValueError('Pressure-level shape mismatch')
        now=float(np.sqrt(np.mean(current[mask]**2)));base=float(np.sqrt(np.mean(reference[mask]**2)))
        pct=float(improvement(np.asarray([now]),np.asarray([base]))[0])
        level_pct=improvement(current,reference)[mask];cells.extend(level_pct.tolist())
        summary[field]=dict(rmse=now,baseline_rmse=base,improvement_percent=pct if np.isfinite(pct) else None,
                            level_counts=counts(level_pct,tolerance))
    return dict(fields=summary,variable_counts=counts([np.nan if v['improvement_percent'] is None else v['improvement_percent'] for v in summary.values()],tolerance),
                level_counts=counts(cells,tolerance),band=band,levels=int(mask.sum()),threshold_percent=tolerance)
