#!/usr/bin/env python3
"""Fit the shared production loss statistics on CPU without a GPU queue wait."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resource-plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from src.models.neuralgcm_residual.numerics import configure_environment
    configure_environment()
    import jax
    from src.models.neuralgcm_residual.backbone import FrozenBackbone
    from src.models.neuralgcm_residual.data import PreparedStore,require_complete_experiment_data
    from src.models.neuralgcm_residual.paper_statistics import fit_statistics
    from src.models.neuralgcm_residual.io import read_json,write_json,sha256
    if any(d.platform!='cpu' for d in jax.devices()):raise RuntimeError('CPU statistics worker expected')
    a.output.mkdir(parents=True,exist_ok=False)
    r=read_json(a.resource_plan)
    b=FrozenBackbone.load(r['checkpoint'],r['checkpoint_sha256'])
    store=PreparedStore(r['prepared']);require_complete_experiment_data(store)
    started=time.perf_counter()
    stats=fit_statistics(b.model,store,60,progress=lambda x:print(json.dumps(x),flush=True))
    stats.update(checkpoint_sha256=r['checkpoint_sha256'],backend='cpu',
        scope='shared production loss statistics for matched K=1/K=2',
        source_sha256=sha256(ROOT/'src/models/neuralgcm_residual/paper_statistics.py'),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    write_json(a.output/'statistics.json',stats,immutable=True)
    print(json.dumps({'statistics_ready':str(a.output/'statistics.json'),'seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':main()
