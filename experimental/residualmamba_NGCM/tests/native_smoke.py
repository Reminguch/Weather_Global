#!/usr/bin/env python3
"""Actual downloaded ERA5 + official NGCM, one pilot chunk (not production T1)."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
from residualmamba_ngcm.artifacts import read_json,write_json,sha256,save_arrays,load_arrays,source_identity
from residualmamba_ngcm.ngcm import load,chunk_baseline
from residualmamba_ngcm.cache import reference_native_forecast
from residualmamba_ngcm.features import static_fields,ancillary,assemble
from residualmamba_ngcm.config import Config
from residualmamba_ngcm.timeline import stack_weather
from src.models.neuralgcm_residual.data import PreparedStore


def main():
    p=argparse.ArgumentParser();p.add_argument('--download-root',type=Path,required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output
    shard=a.download_root/'shards/2015-01/res2p8'
    contract=read_json(shard/'download_contract.json')
    receipts=sorted(shard.glob('train/frames/*.npz.json'))[:29]
    if len(receipts)!=29:raise ValueError('Need 29 completed downloaded frames for the pilot chunk')
    records=[]
    for receipt in receipts:
        record=read_json(receipt);path=(shard/record['file']).resolve()
        if sha256(path)!=record['sha256']:raise ValueError('Corrupt downloaded frame')
        records.append({**record,'file':str(path)})
    manifest={'source':contract['source'],'data_grid':{**contract['grid'],'levels':contract['levels']},
              'records':records,'scope':'partial downloaded pilot; not production'}
    write_json(root/'store/manifest.json',manifest,immutable=True)
    write_json(root/'store/READY.json',{'manifest_sha256':sha256(root/'store/manifest.json')},immutable=True)
    bb,store=load(a.checkpoint,'bdec1b4612c7385fc492aa031db252c66fb74b788a7efbb118b8b60b06644d3e',root/'store',production=False)
    static=static_fields(bb.model)
    print('native chunk, 3 independent steps + native tail of 21',flush=True)
    baseline=chunk_baseline(bb,store,store.times,4)
    save_arrays(root/'pilot_chunk.npz',{'baseline':baseline},{'checkpoint_sha256':bb.checkpoint_sha256,'scope':'pilot'})
    cached,_=load_arrays(root/'pilot_chunk.npz')
    np.testing.assert_array_equal(cached['baseline'],baseline)
    print('independent legacy native evaluation, 21 steps',flush=True)
    reference=reference_native_forecast(bb,store,store.times[7],21)
    std=np.std(np.stack([stack_weather(store.frame(t)) for t in store.times[:8]]).astype(np.float64),axis=(0,3,4))
    # Scale is diagnostic only, never used as production training statistics.
    positive=np.where(std>0,std,1.)
    delta=float(np.max(np.abs(baseline[3:]-reference)/positive[None,:,:,None,None]))
    if not np.isfinite(delta):raise FloatingPointError('Nonfinite baseline comparison')
    inp,forcing=store.inputs_and_forcing(bb.model,store.times[4])
    grid=bb.model.data_coords.horizontal
    e=ancillary(store.times[3:6],forcing,static,grid.latitudes,grid.longitudes)
    if e.shape!=(128,64,19) or not np.isfinite(e).all():raise ValueError('Actual checkpoint ancillary feature mismatch')
    write_json(root/'result.json',{'status':'passed','source_identity':source_identity(),'checkpoint_sha256':bb.checkpoint_sha256,
        'store_identity':store.identity,'shape':list(baseline.shape),'finite':bool(np.isfinite(baseline).all()),
        'cache_roundtrip_bitwise':True,'native_evaluator_max_diff_in_pilot_std':delta,
        'within_reference_bound':delta<=1e-5,'warning':delta>1e-5,'static_shape':list(static.shape),
        'production_T1_3_train_1_val':'pending full restored data and production statistics'},immutable=True)
    print('Native pilot passed',flush=True)

if __name__=='__main__':main()
