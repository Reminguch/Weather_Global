"""One H200 allocation per bounded slice; preserve the full 2k optimizer/controller state."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time


def read(path):return json.loads(Path(path).read_text())


def transition(returncode,before,after,target,budget):
    if returncode not in (0,75):raise RuntimeError('Numerical worker failed: '+str(returncode))
    if not before<after<=target<=budget:raise ValueError('Invalid/no checkpoint progress')
    if returncode==0 and after!=target:raise ValueError('Success without reaching slice target')
    if returncode==75 and after>=target:raise ValueError('Time-boundary exit at/after target')
    return 'complete' if after==budget else 'requeue'


def inspect_checkpoint(output,budget,mode):
    # Called with JAX_PLATFORMS=cpu in the orchestration process. The numerical
    # subprocess retains CUDA. No model or GPU allocation is constructed here.
    from src.models.neuralgcm_residual.io import digest
    from experimental.adaptive_loss.checkpoint import validate
    config=read(output/'config.json')
    identity=digest(config)
    if config['updates']!=budget or config['mode']!=mode:
        raise ValueError('Wrong budget or mode in checkpoint')
    with (output/'last.pkl').open('rb') as f:state=pickle.load(f)
    validate(state,identity,state['params'],state['optimizer'],budget)
    return state,config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--mode',choices=('calibrated','adaptive'),required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    plan=read(a.campaign/'plan.json')
    if plan['hardware']!='H200' or plan['partition']!='ailab':raise ValueError('Expected authorized H200/ailab campaign')
    source=Path(plan['source_root'])
    output=a.campaign/('trial_'+a.mode)
    budget=plan['updates']
    if not 0<plan['slice_seconds']<=3300 or plan['slice_updates']<1:raise ValueError('Invalid bounded slice')
    base=[sys.executable,'-u',str(source/'experimental/weather_k20/train.py'),
          '--resources',plan['resources'],'--statistics',plan['statistics'],
          '--preflight',plan['preflight'],'--lifecycle-report',plan['lifecycle_report'],
          '--mode',a.mode,'--updates',str(budget),'--max-seconds',str(plan['slice_seconds']),
          '--output',str(output)]
    if a.dry_run:
        print(json.dumps(dict(worker=base,slice_updates=plan['slice_updates'],requeue_current_job_only=True),indent=2));return 0
    job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdecimal() or int(job) in plan['a100_jobs_preserved']:
        raise RuntimeError('Requires a dedicated H200 Slurm job; protected A100 IDs cannot be touched')
    lock=(a.campaign/('.'+a.mode+'_driver.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child_env=dict(os.environ,JAX_PLATFORMS='cuda')
    os.environ['JAX_PLATFORMS']='cpu'
    sys.path.insert(0,str(source))
    gate=read(plan['lifecycle_report'])
    if not gate.get('passed') or not gate.get('exact_separate_process_resume'):
        raise ValueError('H200 lifecycle gate has not passed')
    events=a.campaign/(a.mode+'_slice_events.jsonl')
    history=[] if not events.exists() else [json.loads(x) for x in events.read_text().splitlines()]
    def event(**data):
        data.update(job_id=job,restart=os.environ.get('SLURM_RESTART_COUNT','0'),unix_time=time.time())
        with events.open('a') as f:
            f.write(json.dumps(data,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(data),flush=True)
    before=0;resume=(output/'last.pkl').exists()
    if output.exists() and not resume:raise RuntimeError('Incomplete startup artifacts require inspection')
    if resume:
        saved,config=inspect_checkpoint(output,budget,a.mode);before=saved['update']
        if config['source_hashes']!=gate['source_hashes']:raise ValueError('Numerical source provenance differs')
    if (output/'DONE.json').exists():
        done=read(output/'DONE.json')
        if before!=budget or done['identity']!=saved['identity'] or done['updates']!=budget or not done['backbone_frozen']:
            raise ValueError('Invalid completion marker')
        event(event='already_complete',update=before);return 0
    if sum(x['event']=='worker_start' for x in history)>=plan['max_slices']:
        raise RuntimeError('Slice limit reached; preserve checkpoint for review')
    target=min(budget,before+plan['slice_updates'])
    if target<=before:raise RuntimeError('Budget reached without a valid DONE marker')
    command=base+['--stop-after',str(target)]+(['--resume'] if resume else [])
    event(event='worker_start',update=before,target=target,resume=resume)
    result=subprocess.run(command,env=child_env)
    if result.returncode not in (0,75):
        event(event='worker_failure',returncode=result.returncode)
        return result.returncode if result.returncode>0 else 1
    saved,config=inspect_checkpoint(output,budget,a.mode)
    action=transition(result.returncode,before,saved['update'],target,budget)
    if config['source_hashes']!=gate['source_hashes']:raise ValueError('Numerical source provenance differs')
    if action=='complete':
        done=read(output/'DONE.json')
        if done['updates']!=budget or done['identity']!=saved['identity'] or not done['backbone_frozen']:
            raise ValueError('Invalid final completion marker')
        event(event='complete',update=saved['update'],best_update=saved['best']['update']);return 0
    if (output/'DONE.json').exists():raise ValueError('Premature DONE marker')
    event(event='checkpointed_requeue',update=saved['update'],target=target,returncode=result.returncode,
          checkpoint_sha256=hashlib.sha256((output/'last.pkl').read_bytes()).hexdigest())
    # Slurm keeps afterok dependencies pending across requeue of the SAME job.
    # This is the current dedicated H200 job, never any preserved A100 job.
    subprocess.run(['scontrol','requeue',job],check=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
