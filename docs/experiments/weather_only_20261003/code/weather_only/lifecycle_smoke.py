"""Separate GPU processes: uninterrupted versus interrupted adaptive training."""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resources',type=Path,required=True)
    p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--statistics',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    child_env=dict(os.environ,JAX_PLATFORMS='cuda')
    os.environ['JAX_PLATFORMS']='cpu'
    import jax
    import numpy as np
    from src.models.neuralgcm_residual.io import write_json,sha256
    from experimental.weather_only.train import source_hashes
    started=time.monotonic()
    common=[sys.executable,'-u',str(ROOT/'experimental/weather_only/train.py'),
        '--resources',str(a.resources),'--preflight',str(a.preflight),'--statistics',str(a.statistics),
        '--pilot','--mode','adaptive','--updates','6','--validate-every','3','--validation-origins','2',
        '--calibration-batches','1','--window','4','--interval','2','--peak-lr','1e-4','--warmup','4']
    def run(directory,extra,label):
        print(json.dumps(dict(stage=label)),flush=True)
        with (a.output/(label+'.log')).open('w') as f:
            result=subprocess.run(common+['--output',str(directory)]+extra,env=child_env,
                stdout=f,stderr=subprocess.STDOUT,timeout=1000)
        if result.returncode:
            raise RuntimeError(label+' failed:\n'+(a.output/(label+'.log')).read_text()[-6000:])
    continuous,resumed=a.output/'continuous',a.output/'resumed'
    run(continuous,[],'continuous')
    run(resumed,['--stop-after','3'],'interrupted')
    def load(path):
        with path.open('rb') as f:return pickle.load(f)
    stopped=load(resumed/'last.pkl')
    assert stopped['update']==3 and stopped['controller']['last_probe']==2
    assert not (resumed/'DONE.json').exists()
    # Deliberately corrupt ONLY checkpoint-derived views, not the authority.
    for name in ('metrics.jsonl','validation.jsonl'):
        with (resumed/name).open('ab') as f:f.write(b'{"update":999,"loss":0}\n{"update":')
    (resumed/'best.pkl').write_bytes(pickle.dumps(dict(update=999)))
    run(resumed,['--resume'],'resumed')
    left,right=load(continuous/'last.pkl'),load(resumed/'last.pkl')
    for name in ('params','optimizer','rng','best'):
        a_tree,b_tree=left[name],right[name]
        assert jax.tree_util.tree_structure(a_tree)==jax.tree_util.tree_structure(b_tree)
        for x,y in zip(jax.tree_util.tree_leaves(a_tree),jax.tree_util.tree_leaves(b_tree),strict=True):
            np.testing.assert_array_equal(x,y,err_msg='Separate-process restart: '+name)
    for name in ('identity','update','controller','calibration','baseline','validations'):
        assert left[name]==right[name],name
    for name in ('metrics.jsonl','validation.jsonl'):
        def rows(directory):
            result=[json.loads(x) for x in (directory/name).read_text().splitlines()]
            for row in result:row.pop('seconds',None)
            return result
        assert rows(continuous)==rows(resumed),name
    assert left['update']==6 and left['controller']['adjustments']==3
    assert not np.allclose(left['controller']['weights'],1)
    assert load(resumed/'best.pkl')['update']==right['best']['update']
    # Zero-step baseline participates even if every trained model is worse.
    assert left['validations'][0]['update']==0 and left['best']['selection_score']<=1.
    report=dict(passed=True,pilot=True,K=2,updates=6,interruption_update=3,
        exact_separate_process_resume=True,controller_history_exact=True,
        torn_log_and_future_best_recovery=True,initial_candidate_included=True,
        source_hashes=source_hashes(),resources_sha256=sha256(a.resources),
        statistics_sha256=sha256(a.statistics),preflight_sha256=sha256(a.preflight),
        slurm_job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-started)
    write_json(a.output/'report.json',report,immutable=True)
    print(json.dumps(dict(passed=True,seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
