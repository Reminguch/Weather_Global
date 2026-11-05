"""Authenticate unchanged tail1 raw inputs for the five-field v2 diagnostic."""
from pathlib import Path
import numpy as np
from residualmamba_ngcm import ROOT
from residualmamba_ngcm.artifacts import read_json,sha256,versions
from residualmamba_ngcm.reuse import V1_SOURCE


def verify_donor(donor,manifest):
    donor=Path(donor).resolve();plan=read_json(donor/'plan.json')
    cache=donor/'cache';contract=read_json(cache/'contract.json');ready=read_json(cache/'READY.json')
    extension=sha256(ROOT/'experimental/residualmamba_NGCM/tail1_extension.py')
    expected={'ar_tail_k':1,'truth_prefix_steps':23,'source_identity':V1_SOURCE,
              'extension_sha256':extension,'store_identity':manifest['store_identity'],
              'checkpoint_sha256':manifest['checkpoint_sha256'],'libraries':versions()}
    if any(contract.get(k)!=v for k,v in expected.items()):raise ValueError('Tail1 producer contract mismatch')
    if contract['plan_sha256']!=sha256(donor/'plan.json') or ready['contract_sha256']!=sha256(cache/'contract.json'):
        raise ValueError('Tail1 plan/READY mismatch')
    main=Path(manifest['reused_inputs']['donor_root'])
    if any(plan['parent_artifacts'][n]!=sha256(main/n) for n in ('stats.npz','geometry.npz','setup.json')):
        raise ValueError('Tail1 must share the same train-only statistics and grid')
    if set(ready['chunks'])!=set(read_json(main/'cache/READY.json')['chunks']) or len(ready['chunks'])!=484:
        raise ValueError('Tail1 cache is incomplete')
    for anchor,value in ready['chunks'].items():
        meta=read_json(cache/f'chunk_{int(anchor):05d}.json')
        if meta['sha256']!=value or any(meta.get(k)!=v for k,v in contract.items()):
            raise ValueError('Tail1 chunk receipt mismatch')
    qa=read_json(donor/'qa/COMPLETE.json')
    checks=read_json(donor/'cache-verified.json')
    if qa['source_identity']!=V1_SOURCE or not qa['finite'] or len(checks['records'])!=4:
        raise ValueError('Tail1 native/restart evidence missing')
    if any(not np.isfinite(row[k]) for row in checks['records']
           for k in ('prefix_max_diff_in_std','tail_max_diff_in_std')):
        raise ValueError('Nonfinite tail1 native verification')
    files=('plan.json','cache/contract.json','cache/READY.json','cache-verified.json','qa/COMPLETE.json')
    return {'donor_root':str(donor),'cache':str(cache),'files':{n:sha256(donor/n) for n in files},
            'note':'Reused native cache/loader evidence; v2 core qualification is separate. No v1 trained parameters imported.'}


def check_receipts(receipt):
    root=Path(receipt['donor_root'])
    if any(sha256(root/n)!=v for n,v in receipt['files'].items()):
        raise ValueError('Tail1 reused inputs changed')
