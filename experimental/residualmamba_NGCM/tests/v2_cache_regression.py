#!/usr/bin/env python3
"""Recompute the v2 zero-head census from all 60 existing validation chunks."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from residualmamba_ngcm.artifacts import load_arrays, read_json, write_json, sha256, source_identity
from residualmamba_ngcm.cache import validate_chunk, cache_path
from residualmamba_ngcm.config import SCORED_FIELDS
from residualmamba_ngcm.loss import target_norm, contributions
from residualmamba_ngcm.timeline import split_segments, stack_weather
from src.models.neuralgcm_residual.data import PreparedStore


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--donor-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args(); root=args.donor_root
    plan=read_json(root/'plan.json'); setup=read_json(root/'setup.json')
    store=PreparedStore(Path(plan['download_root'])/'res2p8')
    stats,sm=load_arrays(root/'stats.npz'); geometry,_=load_arrays(root/'geometry.npz')
    contract=read_json(root/'cache/contract.json'); ready=read_json(root/'cache/READY.json')
    assert sm['store_identity']==setup['store_identity']==store.identity==contract['store_identity']
    assert sm['preflight_hash']==sha256(root/'setup.json')
    assert ready['contract_hash']==sha256(root/'cache/contract.json')
    totals=np.zeros((24,5,29),np.float64); n=0
    for segment in split_segments(store.times,'val')[0]:
        for off in range(0,96,24):
            a0=int(segment[off]); arrays,meta=load_arrays(cache_path(root/'cache',a0))
            assert meta['sha256']==ready['chunks'][str(a0)]
            assert all(meta[k]==v for k,v in contract.items())
            baseline=arrays['baseline']; validate_chunk(baseline,meta,store.times,a0)
            for i in range(24):
                truth=stack_weather(store.frame(store.times[a0+i+1]))
                target=target_norm(truth,baseline[i],stats)
                totals[i]+=np.asarray(contributions(np.zeros_like(target),target,geometry['latitude']))
            n+=1
            print(f'v2 zero-head census {n}/60',flush=True)
    assert n==60
    totals/=n
    reference=np.asarray(read_json(root/'census.json')['loss_by_step_field_level'])[:,:5]
    np.testing.assert_allclose(totals,reference,rtol=1e-6,atol=1e-10)
    fields=totals.mean(axis=0).sum(axis=1); value=float(fields.sum())
    np.testing.assert_allclose(value,3.525165,rtol=1e-6)
    shares=totals.mean(axis=0)/value
    hot=[{'field':SCORED_FIELDS[v],'level_index':int(p),'share':float(shares[v,p])}
         for v,p in zip(*np.where(shares>.1))]
    write_json(args.output,{'status':'passed','source_identity':source_identity(),
        'chunks':n,'mean_loss':value,'per_field':dict(zip(SCORED_FIELDS,fields.tolist())),
        'field_share':dict(zip(SCORED_FIELDS,(fields/value).tolist())),
        'max_abs_difference_from_v1_five_fields':float(np.max(np.abs(totals-reference))),
        'loss_by_step_field_level':totals.tolist(),'above_ten_percent':hot,
        'stats_sha256':sha256(root/'stats.npz'),'cache_ready_sha256':sha256(root/'cache/READY.json'),
        'v1_census_sha256':sha256(root/'census.json')},immutable=True)
    print(f'v2 regression passed: {value:.10f}',flush=True)


if __name__=='__main__':main()
