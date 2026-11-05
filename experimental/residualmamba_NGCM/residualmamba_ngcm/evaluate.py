"""Native open-loop cold/warm forecasts, controls and paired Gaussian metrics."""
from pathlib import Path
import jax
import numpy as np
from .config import SCORED_FIELDS,LEVELS
from .timeline import stack_weather,evaluation_origins
from .features import ancillary,assemble,static_fields
from .model import zero_memory
from .ngcm import rollout
from .loss import correct,target_norm,weighted_mse,contributions
from .artifacts import save_arrays,write_json,read_json,sha256,load_arrays
from src.models.neuralgcm_residual.data import STEP,split_for
from src.models.neuralgcm_residual.loss import gaussian_weights


def parity_origins(manifest_path,eval_times_path,gc_origins_path):
    """Accept actual GC dataset times and logged origins, never guessed offsets."""
    manifest=read_json(manifest_path)
    indices=np.asarray(manifest['anchor_indices'],dtype=np.int64)
    times=np.load(eval_times_path,allow_pickle=False).astype('datetime64[h]')
    logged=np.asarray(read_json(gc_origins_path)['origins'],dtype='datetime64[h]')
    if indices.shape!=(32,) or len(set(indices))!=32 or np.any(indices<0) or np.any(indices>=len(times)):
        raise ValueError('E1 requires the 32 unique GC final-input indices')
    origins=times[indices]
    np.testing.assert_array_equal(origins,logged)
    if any(str(t)[:4]!='2022' for t in origins):
        raise ValueError('E1 must contain 2022 origins')
    return origins


def forecast(bb,store,model,params,initial_memory,stats,config,origin,warm=False,steps=40,bias=None):
    grid=bb.model.data_coords.horizontal
    static=static_fields(bb.model)
    origin=np.datetime64(origin,'h')
    memory=zero_memory(initial_memory)
    apply=jax.jit(model.apply) if model is not None else None
    key=jax.random.PRNGKey(config.seed)
    if warm and model is not None:
        required=[origin+k*STEP for k in range(-7,1)]
        if any(t not in store.times or split_for(t)!=split_for(origin) or str(t)[:4]!=str(origin)[:4] for t in required):
            raise ValueError('Warm-3 history crosses the evaluation split year')
        for t in [origin-3*STEP,origin-2*STEP,origin-STEP]:
            b,forcing=next(rollout(bb,store,t,1))
            older,newer=stack_weather(store.frame(t-STEP)),stack_weather(store.frame(t))
            extra=ancillary([t-STEP,t,t+STEP],forcing,static,grid.latitudes,grid.longitudes)
            features=assemble(older,newer,b,extra,stats,config,6)
            if config.reset_every_step:memory=zero_memory(memory)
            key,subkey=jax.random.split(key)
            _,memory=apply(params,memory,subkey,features)
    prev,cur=stack_weather(store.frame(origin-STEP)),stack_weather(store.frame(origin))
    for k,(baseline,forcing) in enumerate(rollout(bb,store,origin,steps),1):
        target_time=origin+k*STEP
        target=stack_weather(store.frame(target_time))
        if bias is not None:
            corrected=baseline.copy()
            if bias[k-1].shape != (5,29,*baseline.shape[-2:]):
                raise ValueError('v2 bias must contain only five fields and 29 levels')
            corrected[:5,8:]+=bias[k-1]
        elif model is None:
            corrected=baseline.copy()
        else:
            extra=ancillary([target_time-2*STEP,target_time-STEP,target_time],forcing,static,grid.latitudes,grid.longitudes)
            features=assemble(prev,cur,baseline,extra,stats,config,6*k)
            if config.reset_every_step:memory=zero_memory(memory)
            key,subkey=jax.random.split(key)
            out,memory=apply(params,memory,subkey,features)
            corrected=np.asarray(correct(baseline,out,stats))
        if not np.isfinite(corrected).all():
            raise FloatingPointError(f'Nonfinite corrected forecast: {origin} + {6*k}h')
        yield k,baseline,corrected,target
        prev,cur=cur,baseline  # Critical: corrected fields never become inputs.


def metrics(prediction,target,climatology,latitude):
    w=gaussian_weights(latitude).astype(np.float64)
    area=lambda x:np.sum(x*w,axis=-1).mean(axis=-1)
    p,t,c=(np.asarray(v,np.float64) for v in (prediction,target,climatology))
    mse=area((p-t)**2)
    numerator=area((p-c)*(t-c))
    denominator=np.sqrt(area((p-c)**2)*area((t-c)**2))
    acc=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
    return mse,acc


def spectra(grid,arrays):
    result=[]
    for field,level in ((1,500),(0,850),(2,850)):
        p=LEVELS.index(level)
        result.append([np.sum(np.asarray(grid.to_modal(a[field,p]))**2*np.asarray(grid.mask),axis=-2)
                       for a in arrays])
    return np.asarray(result)


def bootstrap(origins,corrected,baseline,repeats=1000):
    """Paired calendar-week blocks; resample origins, never independent leads."""
    week=np.asarray(origins,dtype='datetime64[D]').astype(np.int64)//7
    blocks=[np.flatnonzero(week==v) for v in np.unique(week)]
    rng=np.random.default_rng(22)
    estimates=[]
    for _ in range(repeats):
        index=np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))])
        c,b=corrected[index].mean(axis=0),baseline[index].mean(axis=0)
        estimates.append(np.divide(100*(b-c),b,out=np.full_like(b,np.nan),where=b>0))
    return np.nanpercentile(estimates,[2.5,97.5],axis=0)


def run(bb,store,model,params,initial_memory,stats,config,origins,root,identity,warm=False,bias=None):
    root=Path(root)
    grid=bb.model.data_coords.horizontal
    skipped=[]
    used=[]
    for origin in origins:
        if warm and any(str(origin+k*STEP)[:4]!=str(origin)[:4] or origin+k*STEP not in store.times for k in range(-7,1)):
            skipped.append(str(origin));continue
        path=root/f'origin_{str(origin).replace(":", "")}.npz'
        if path.exists():
            _,meta=load_arrays(path)
            if meta['identity']!=identity or meta['warm']!=warm:
                raise ValueError('Evaluation artifact belongs to a different run')
            used.append(origin);continue
        mses,accs,js,ps,jfields=[],[],[],[],[]
        for k,b,y,t in forecast(bb,store,model,params,initial_memory,stats,config,origin,warm,bias=bias):
            bm,ba=metrics(b[:5],t[:5],stats['climatology'][:5],grid.latitudes)
            ym,ya=metrics(y[:5],t[:5],stats['climatology'][:5],grid.latitudes)
            mses.append([bm,ym]);accs.append([ba,ya])
            r=target_norm(t,b,stats)
            correction=(y[:5,8:]-b[:5,8:])/stats['diff_std'][:5,8:,None,None]
            fields=np.stack([np.asarray(contributions(c,r,grid.latitudes)).sum(axis=-1)
                             for c in (np.zeros_like(r),correction)])
            jfields.append(fields);js.append(fields.sum(axis=-1))
            if k in (4,20,40):ps.append(spectra(grid,[t,b,y]))
        save_arrays(path,{'mse':np.asarray(mses),'acc':np.asarray(accs),'J':np.asarray(js),'J_per_field':np.asarray(jfields),'spectra':np.asarray(ps)},
            {'origin':str(origin),'identity':identity,'warm':warm,'field_order':SCORED_FIELDS,'levels':LEVELS,
             'model_axis':['native','corrected'],'spectrum_models':['truth','native','corrected'],
             'spectrum_fields':['Z500','T850','U850'],'spectrum_leads_hours':[24,120,240]})
        used.append(origin)
        print(f'evaluated {origin} warm={warm}',flush=True)
    if not used:raise ValueError('No eligible evaluation origins')
    rows=[load_arrays(root/f'origin_{str(o).replace(":", "")}.npz')[0] for o in used]
    mse=np.stack([r['mse'] for r in rows]); j=np.stack([r['J'] for r in rows])
    base,corrected=mse[:,:,0].mean(axis=0),mse[:,:,1].mean(axis=0)
    change=np.divide(100*(base-corrected),base,out=np.full_like(base,np.nan),where=base>0)
    jf=np.mean(np.stack([r['J_per_field'] for r in rows]),axis=0)
    jf_share=np.divide(jf,jf.sum(axis=-1,keepdims=True),out=np.zeros_like(jf),where=jf.sum(axis=-1,keepdims=True)>0)
    save_arrays(root/'summary.npz',{'origins':np.asarray(used,dtype='datetime64[h]'),
        'rmse_native':np.sqrt(base),'rmse_corrected':np.sqrt(corrected),'mse_improvement_pct':change,
        'mse_improvement_ci95':bootstrap(used,mse[:,:,1],mse[:,:,0]),
        'J_native':j[:,:,0].mean(axis=0),'J_corrected':j[:,:,1].mean(axis=0),
        'J_per_field':jf,'J_field_share':jf_share,
        'J_improvement_ci95':bootstrap(used,j[:,:,1],j[:,:,0]),
        'acc':np.nanmean(np.stack([r['acc'] for r in rows]),axis=0),
        'spectra':np.mean(np.stack([r['spectra'] for r in rows]),axis=0)},
        {'identity':identity,'field_order':SCORED_FIELDS,'warm':warm,'skipped_origins':skipped,'origins':len(used),
         'bootstrap':'1000 paired calendar-week blocks','historical_2022_tuning':True,
         'claim':'2023 held-out only; 2022 has historical tuning exposure'})


def bias_table(bb,store,path,upstream,stats=None,audit_root=None):
    """C1 streaming mean over ALL eligible 00/12 UTC train origins, never validation."""
    candidates=[t for t in store.times if split_for(t)=='train' and int(t.astype(np.int64))%12==0
        and all(t+k*STEP in store.times and split_for(t+k*STEP)=='train' for k in range(-4,41))]
    mean=np.zeros((40,5,29,128,64),dtype=np.float64)
    if not candidates:raise ValueError('No C1 training origins')
    for n,origin in enumerate(candidates,1):
        values=[]
        for k,(b,_) in enumerate(rollout(bb,store,origin,40)):
            residual=stack_weather(store.frame(origin+(k+1)*STEP))[:5,8:]-b[:5,8:]
            mean[k]+=(residual-mean[k])/n
            if stats is not None:
                r=residual/stats['diff_std'][:5,8:,None,None]
                values.append(float(weighted_mse(np.zeros_like(r),r,bb.model.data_coords.horizontal.latitudes)))
        if stats is not None and audit_root is not None:
            write_json(Path(audit_root)/f'{origin}.json',{'origin':str(origin),'J':values,'upstream':upstream},immutable=True)
        print(f'C1 {n}/{len(candidates)} {origin}',flush=True)
    save_arrays(path,{'bias':mean.astype(np.float32)}, {'upstream':upstream,
        'origins':[str(t) for t in candidates],'train_only':True,'leads_hours':list(range(6,241,6))})


def baseline_audit(bb,store,stats,root,upstream):
    """Year-by-year native J_k, all eligible 00/12 UTC origins, with resumable receipts."""
    root=Path(root)
    years={}
    for year in range(2015,2024):
        # Train origins may continue across calendar years inside the train split.
        origins=[t for t in store.times if int(str(t)[:4])==year and int(t.astype(np.int64))%12==0
            and all(t+k*STEP in store.times and split_for(t+k*STEP)==split_for(t) for k in range(-4,41))]
        scores=[]
        for origin in origins:
            path=root/f'{origin}.json'
            if path.exists():
                record=read_json(path)
                if record['upstream']!=upstream:raise ValueError('Baseline audit provenance changed')
            else:
                values=[]
                for k,(b,_) in enumerate(rollout(bb,store,origin,40),1):
                    truth=stack_weather(store.frame(origin+k*STEP))
                    r=target_norm(truth,b,stats)
                    values.append(float(weighted_mse(np.zeros_like(r),r,bb.model.data_coords.horizontal.latitudes)))
                record={'origin':str(origin),'J':values,'upstream':upstream}
                write_json(path,record,immutable=True)
            scores.append(record['J'])
        if not scores or not np.isfinite(scores).all():raise ValueError(f'Incomplete/nonfinite baseline audit: {year}')
        years[str(year)]={'origins':len(origins),'J':np.asarray(scores).mean(axis=0).tolist()}
    write_json(root/'summary.json',{'upstream':upstream,'years':years,'selection':'all eligible 00/12 UTC'},immutable=True)
