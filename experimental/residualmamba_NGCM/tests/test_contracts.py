"""Small deterministic tests for contracts that previously drifted between trainers."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import haiku as hk
from residualmamba_ngcm.config import Config,LEVELS,P
from residualmamba_ngcm.timeline import (window,origin_index,step_times,split_segments,
    TrainingCursor,advance_cursor,lead_hours,evaluation_origins)
from residualmamba_ngcm.features import (assemble,grid_nodes,from_grid_nodes,channel_names,time_channels)
from residualmamba_ngcm.loss import (weighted_mse,target_norm,correct,weights)
from residualmamba_ngcm.stats import Moments,scale_floors
from residualmamba_ngcm.train import chunk_loss,run
from residualmamba_ngcm.artifacts import load_checkpoint,save_checkpoint,load_arrays,save_arrays
from residualmamba_ngcm.evaluate import parity_origins,bootstrap

LAT=np.arcsin(np.polynomial.legendre.leggauss(4)[0])


def test_strict_protocol_and_controls():
    cfg=Config()
    assert cfg.channels==796 and cfg.v24().adam_beta2==.98
    assert len(cfg.swa_steps)==13
    assert Config(arm='C4').channels==537 and Config(arm='C3').channels==797
    with pytest.raises(ValueError):replace(cfg,bptt_steps=2)
    with pytest.raises(TypeError):Config(unknown=True)


def test_t3_t4_t5_timeline_and_future_leakage():
    # Independent GC reference: append only the baseline after prefix-1.
    truth=np.arange(80,dtype=np.float32)
    a0=9
    frames=list(truth[a0-1:a0+4]);baselines=[];reference=[]
    for i in range(24):
        pair=(frames[i],frames[i+1]);pred=.1*pair[0]+.7*pair[1]
        reference.append(pair);baselines.append(pred)
        if 3<=i<23:frames.append(pred)
    prefix=truth[a0-1:a0+4]
    for i in range(24):np.testing.assert_array_equal(window(prefix,baselines,i),reference[i])
    mutated=truth.copy();mutated[a0+4:]+=100000
    for i in range(24):
        np.testing.assert_array_equal(window(mutated[a0-1:a0+4],baselines,i),reference[i])
    times=np.arange(np.datetime64('2020-01-01T00'),np.datetime64('2020-02-01T00'),np.timedelta64(6,'h'))
    for i in range(24):
        assert origin_index(a0,i)==a0+min(i,3)
        np.testing.assert_array_equal(step_times(times,a0,i),times[[a0+i-1,a0+i,a0+i+1]])
    assert [lead_hours(i) for i in range(24)]==[6,6,6]+list(range(6,127,6))


def test_segments_and_t9_cursor_reset():
    times=np.arange(np.datetime64('2015-01-01T00'),np.datetime64('2024-01-01T00'),np.timedelta64(6,'h'))
    train,omitted=split_segments(times,'train');val,omitted_val=split_segments(times,'val')
    assert len(train)==106 and len(val)==15
    assert omitted==47 and omitted_val==15
    cursor=TrainingCursor();cfg=Config().v24()
    assert [int(train[0][o]) for o in (0,24,48,72)]==[4,28,52,76]
    for _ in range(4):cursor=advance_cursor(cursor,cfg,len(train))
    assert cursor.segment_offset==0 and cursor.segment_index==1
    cursor=advance_cursor(TrainingCursor(0,105,72),cfg,106)
    assert cursor==TrainingCursor(1,0,0)


def hand_stats():
    return {'mean':np.full((7,37),10,np.float32),'std':np.full((7,37),2,np.float32),
            'diff_std':np.full((7,37),4,np.float32),'extra_mean':np.ones(19,np.float32),
            'extra_std':np.full(19,2,np.float32)}


def test_t7_normalization_and_control_feature_layout():
    x=np.full((7,37,2,4),14,np.float32);extra=np.full((2,4,19),5,np.float32)
    stats=hand_stats();cfg=Config()
    f=assemble(x,x,x,extra,stats,cfg)
    assert f.shape==(2,4,796)
    np.testing.assert_array_equal(f,2)
    np.testing.assert_array_equal(target_norm(x+12,x,stats),3)
    np.testing.assert_array_equal(np.asarray(correct(x,np.ones((7,29,2,4)),stats))[:,8:],18)
    assert len(set(channel_names()))==796
    no_next=assemble(x,x,x+900,extra,stats,Config(arm='C4'))
    assert no_next.shape[-1]==537 and np.all(no_next==2)
    lead=assemble(x,x,x,extra,stats,Config(arm='C3'),126)
    np.testing.assert_allclose(lead[...,-1],126/240)


def test_t11_orientation():
    x=np.broadcast_to(np.sin(LAT)[None,:,None],(8,4,3)).copy()
    nodes=np.asarray(grid_nodes(x))
    np.testing.assert_allclose(nodes[:,0,0],np.repeat(np.sin(LAT),8),rtol=1e-7)
    np.testing.assert_array_equal(np.asarray(from_grid_nodes(nodes,8,4)),x.astype(np.float32))


def test_t6_t12_t13_weights_zero_head_scatter():
    lat,p,alpha=weights(LAT)
    np.testing.assert_allclose(np.mean(lat),1,atol=1e-7)
    assert np.mean(P)==15470/29
    ones=np.ones((7,29,8,4),np.float32)
    np.testing.assert_allclose(weighted_mse(ones,np.zeros_like(ones),LAT),5.1,rtol=1e-6)
    b=np.random.default_rng(22).normal(size=(7,37,8,4)).astype(np.float32)
    b[:,:,0,0]=-0.
    y=np.asarray(correct(b,np.zeros_like(ones),hand_stats()))
    np.testing.assert_array_equal(y.view(np.uint32),b.view(np.uint32))
    y=np.asarray(correct(b,ones,hand_stats()))
    np.testing.assert_array_equal(y[:,:8].view(np.uint32),b[:,:8].view(np.uint32))
    from src.models.neuralgcm_residual.loss import gaussian_weights
    with pytest.raises(ValueError):gaussian_weights(np.linspace(-1,1,4))


def test_t14_statistics_population_cloud_floors():
    x=np.array([[1,2,4,8],[3,8,9,10]],np.float64)
    moments=Moments();moments.add(x[:,:2]);moments.add(x[:,2:])
    mean,std=moments.result()
    np.testing.assert_allclose(mean,x.mean(-1));np.testing.assert_allclose(std,x.std(-1))
    scale=np.ones((7,37));scale[5,0]=0;scale[6,4]=1e-6
    result,record=scale_floors(scale)
    assert result[5,0]==np.float32(.001) and record[0]['levels_hpa']==[1]
    assert record[1]['levels_hpa']==[7]
    scale[4,0]=0
    with pytest.raises(ValueError):scale_floors(scale)


def test_progress_features_use_graphcast_helpers():
    times=np.array(['2020-01-01T00','2020-01-01T06','2020-01-01T12'],dtype='datetime64[h]')
    x=time_channels(times,LAT,np.arange(8)*2*np.pi/8)
    assert x.shape==(8,4,15) and np.isfinite(x).all()
    # Local day phase differs with longitude; year phase does not.
    np.testing.assert_array_equal(x[0,:,3:9],x[1,:,3:9])
    assert not np.array_equal(x[0,:,9:],x[1,:,9:])


def test_t8_scan_remat_matches_explicit_reverse_vjp_and_stops_boundary():
    features=jnp.linspace(.1,.9,24)
    targets=jnp.ones((24,7,29,1,4))*.3
    def apply(p,state,key,x):
        h=jnp.tanh(p['w']*state['h']+x)
        return jnp.broadcast_to(p['v']*h,(7,29,1,4)),{'h':h}
    p={'w':jnp.array(.6),'v':jnp.array(.2)};state={'h':jnp.array(.9)}
    batch={'features':features,'target':targets}
    f=lambda p,s:chunk_loss(apply,p,s,batch,jax.random.PRNGKey(22),LAT)
    (_,end),scan_grad=jax.value_and_grad(f,has_aux=True)(p,state)
    boundary=jax.grad(lambda s:f(p,s)[0])(state)
    assert float(boundary['h'])==0 and float(end['h'])!=0
    tape=[];memory=state
    for x,t in zip(features,targets):
        def step(p,m):
            y,m=apply(p,m,None,x)
            return m,weighted_mse(y,t,LAT)/24
        (next_memory,_),pullback=jax.vjp(step,p,memory)
        tape.append(pullback);memory=next_memory
    g=jax.tree_util.tree_map(jnp.zeros_like,p);gh=jax.tree_util.tree_map(jnp.zeros_like,memory)
    for reverse in reversed(tape):
        gp,gh=reverse((gh,jnp.array(1.)))
        g=jax.tree_util.tree_map(lambda a,b:a+b,g,gp)
    for a,b in zip(jax.tree_util.tree_leaves(g),jax.tree_util.tree_leaves(scan_grad)):
        np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-7)


def test_checkpoint_integrity_and_array_receipts(tmp_path):
    path=tmp_path/'checkpoint.pkl'
    save_checkpoint(path,{'step':2,'identity':'x','params':{'a':np.array(2.)}})
    assert load_checkpoint(path,'x')['step']==2
    with pytest.raises(ValueError):load_checkpoint(path,'wrong')
    with pytest.raises(FileExistsError):save_checkpoint(path,{})
    path.write_bytes(path.read_bytes()+b'corruption')
    with pytest.raises(ValueError):load_checkpoint(path)
    path=tmp_path/'data.npz';save_arrays(path,{'x':np.ones(3)},{'kind':'test'})
    assert load_arrays(path)[0]['x'].sum()==3


def test_exact_training_resume_and_segment_carry(tmp_path):
    def forward(x):
        p=hk.get_parameter('p',(),init=hk.initializers.Constant(.2))
        h=hk.get_state('h',(),init=hk.initializers.Constant(0))
        new=jnp.tanh(h*p+x)
        hk.set_state('h',new)
        return jnp.broadcast_to(new*p,(7,29,1,4))
    model=hk.transform_with_state(forward)
    segments=(np.arange(96),np.arange(96,192))
    batch={'features':np.linspace(.1,.5,24,dtype=np.float32),
           'target':np.ones((24,7,29,1,4),np.float32)*.1}
    load=lambda a:batch
    cfg=Config();full=tmp_path/'full';split=tmp_path/'split'
    run(model,cfg,segments,(),load,LAT,full,'test',steps=5)
    run(model,cfg,segments,(),load,LAT,split,'test',steps=2)
    run(model,cfg,segments,(),load,LAT,split,'test',steps=5,resume=split/'step_00002.pkl')
    a=load_checkpoint(full/'step_00005.pkl');b=load_checkpoint(split/'step_00005.pkl')
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(x,y)
    assert a['cursor']=={'epoch':0,'segment_index':1,'segment_offset':24}


def test_evaluation_origins_and_paired_bootstrap():
    times=np.arange(np.datetime64('2023-01-01T00'),np.datetime64('2024-01-01T00'),np.timedelta64(6,'h'))
    cold=evaluation_origins(times,2023);warm=evaluation_origins(times,2023,warm=True)
    assert len(cold)>700 and set(warm)<set(cold)
    assert all(int(t.astype(np.int64))%12==0 for t in cold)
    b=np.arange(1,len(cold)+1,dtype=float)[:,None]
    ci=bootstrap(cold,.8*b,b,repeats=20)
    np.testing.assert_allclose(ci,20,rtol=1e-10)


def test_t5_actual_v24_iterator_parity():
    import xarray as xr
    import pandas as pd
    from src.models.mamba.v24_Ilya.baseline_cache import iter_trajectory
    def frame(x):
        return xr.Dataset({'temperature':(('batch','time'),np.array([[x]],np.float32))},
                          coords={'batch':[0],'time':[np.timedelta64(0,'h')]})
    prefix=[frame(i) for i in range(5)]
    chunk=SimpleNamespace(input_frames=prefix,static_inputs=xr.Dataset(),
                          truths=[frame(i+2) for i in range(24)],forcings=[xr.Dataset(coords={'time':[np.timedelta64(6,'h')]}) for _ in range(24)])
    def predict(inputs,targets,forcing):
        x=np.asarray(inputs.temperature).reshape(2)
        return frame(.1*x[0]+.7*x[1])
    values=list(iter_trajectory(chunk,predict,4,pd.Timedelta(hours=6)))
    baseline=[float(pred.temperature.values.item()) for _,_,pred in values]
    for i,inputs,_ in values:
        np.testing.assert_allclose(window(np.arange(5),baseline,i),inputs.temperature.values.reshape(2),rtol=1e-6)


def test_t1_t2_t3_t4_fake_native_rollout_and_reference():
    from residualmamba_ngcm.ngcm import chunk_baseline
    from residualmamba_ngcm.cache import reference_native_forecast
    from residualmamba_ngcm.config import FIELDS
    class Store:
        times=np.arange(np.datetime64('2020-01-01T00'),np.datetime64('2020-02-01T00'),np.timedelta64(6,'h'))
        future_change=0.
        def frame(self,t):
            i=int((t-self.times[0])/np.timedelta64(6,'h'))
            return {f:np.full((37,2,4),float(i)+(self.future_change if i>=8 else 0),np.float32) for f in FIELDS}
        def inputs_and_forcing(self,model,t):return self.frame(t),{'origin':int((t-self.times[0])/np.timedelta64(6,'h'))}
    class Backbone:
        model=None
        def encode(self,x,f):return {'time':f['origin'],'value':float(x[FIELDS[0]][0,0,0]),'origin':f['origin']}
        def advance_6h(self,s,f):
            assert s['origin']==f['origin']
            return {**s,'time':s['time']+1,'value':s['value']*.9+f['origin']*.01}
        def assert_time(self,a,b,k):assert b['time']==a['time']+k
        def decode(self,s,f):
            assert s['origin']==f['origin']
            return {v:np.full((37,2,4),s['value'],np.float32) for v in FIELDS}
    store=Store();bb=Backbone()
    b=chunk_baseline(bb,store,store.times,4)
    tail=reference_native_forecast(bb,store,store.times[7],21)
    np.testing.assert_array_equal(b[3:],tail)
    store.future_change=1000
    np.testing.assert_array_equal(chunk_baseline(bb,store,store.times,4),b)


def test_parity_origins_requires_real_gc_timestamps(tmp_path):
    import json
    times=np.arange(np.datetime64('2022-01-01T00'),np.datetime64('2022-02-01T00'),np.timedelta64(6,'h'))
    idx=np.arange(9,41)
    manifest=tmp_path/'anchors.json';logged=tmp_path/'logged.json';grid=tmp_path/'times.npy'
    manifest.write_text(json.dumps({'anchor_indices':idx.tolist()}));np.save(grid,times)
    logged.write_text(json.dumps({'origins':[str(t) for t in times[idx]]}))
    np.testing.assert_array_equal(parity_origins(manifest,grid,logged),times[idx])
    logged.write_text(json.dumps({'origins':[str(t+np.timedelta64(6,'h')) for t in times[idx]]}))
    with pytest.raises(AssertionError):parity_origins(manifest,grid,logged)


def test_census_exposes_dominant_field_without_aborting_finite_metrics(tmp_path):
    from residualmamba_ngcm.train import census
    batch={'target':np.zeros((24,7,29,1,4),np.float32)}
    batch['target'][:,4,2]=100
    result=census((np.arange(96),),lambda _:batch,LAT,tmp_path/'census.json',{'test':'mock'})
    assert result['decision_required']
    assert result['above_ten_percent'][0]['field_index']==4
    np.testing.assert_allclose(result['above_ten_percent'][0]['share'],1)


def test_eval_never_feeds_corrected_weather_back(monkeypatch):
    from residualmamba_ngcm import evaluate
    from residualmamba_ngcm.config import FIELDS
    times=np.arange(np.datetime64('2023-02-01T00'),np.datetime64('2023-03-01T00'),np.timedelta64(6,'h'))
    class Store:
        def frame(self,t):
            value=float((t-times[0])/np.timedelta64(6,'h'))
            return {f:np.full((37,2,4),value,np.float32) for f in FIELDS}
        def inputs_and_forcing(self,model,t):
            return self.frame(t),{'sea_surface_temperature':np.ones((1,2,4),np.float32),
                                  'sea_ice_cover':np.zeros((1,2,4),np.float32)}
    store=Store();store.times=times
    class BB:
        model=SimpleNamespace(data_coords=SimpleNamespace(horizontal=SimpleNamespace(latitudes=LAT,longitudes=np.array([0.,np.pi]))))
        def encode(self,inputs,forcing):return (0,float(inputs[FIELDS[0]][0,0,0]))
        def advance_6h(self,z,forcing):return (z[0]+1,z[1]*.9)
        def assert_time(self,a,b,k):assert b[0]==a[0]+k
        def decode(self,z,f):return {v:np.full((37,2,4),z[1],np.float32) for v in FIELDS}
    monkeypatch.setattr(evaluate,'static_fields',lambda _:np.zeros((2,4,2),np.float32))
    seen=[]
    class Model:
        def apply(self,p,h,key,x):
            # A visible huge correction would immediately expose accidental feedback.
            return jnp.full((7,29,2,4),1000.),{'h':h['h']+1}
    stats=hand_stats();stats['mean'][:]=0;stats['std'][:]=1;stats['diff_std'][:]=1
    baseline=list(evaluate.forecast(BB(),store,None,None,{},stats,Config(),times[10],steps=3))
    corrected=list(evaluate.forecast(BB(),store,Model(),{}, {'h':jnp.array(0.)},stats,Config(),times[10],steps=3))
    for (_,b,y,t),(_,ref,_,_) in zip(corrected,baseline):
        np.testing.assert_array_equal(b,ref)
        np.testing.assert_array_equal(y[:,:8],b[:,:8])
        np.testing.assert_allclose(y[:,8:]-b[:,8:],1000,rtol=1e-6)


def test_census_zero_error_step_class_is_valid(tmp_path):
    from residualmamba_ngcm.train import census
    batch={'target':np.zeros((24,7,29,1,4),np.float32)}
    batch['target'][3,0,0]=1
    result=census((np.arange(96),),lambda _:batch,LAT,tmp_path/'census-zero.json',{})
    assert np.isfinite(result['step_class_share']).all()
    np.testing.assert_array_equal(result['step_class_share'][0],0)


def test_swa_uses_fixed_thirteen_snapshots_and_only_parameters(tmp_path):
    from residualmamba_ngcm.swa import average
    cfg=Config()
    for step in cfg.swa_steps:
        save_checkpoint(tmp_path/f'step_{step:05d}.pkl',{'step':step,'identity':'same-run',
            'params':{'weight':np.array([step],np.float32)},'memory':{'h':np.array(step*9)},'opt_state':step})
    save_checkpoint(tmp_path/'step_10000.pkl',{'step':10000,'identity':'same-run','params':{'weight':np.array([1e9],np.float32)}})
    output=tmp_path/'swa.pkl';average(tmp_path,cfg,output)
    result=load_checkpoint(output,'same-run')
    np.testing.assert_array_equal(result['params']['weight'],5000)
    assert 'memory' not in result and 'opt_state' not in result
    assert len(result['source_checkpoints'])==13 and result['kind']=='fixed_swa'
