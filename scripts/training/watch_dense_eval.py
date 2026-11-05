#!/usr/bin/env python3
"""One concurrent H200 evaluator, discovering atomic training checkpoints."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
from ngcm_2k_controller import state_of,ACTIVE,RETRYABLE,batch,read,write
REPO=Path(__file__).resolve().parents[2]


def candidates(root,plan):
    result=[]
    for arm,run in plan['arms'].items():
        for path in sorted((Path(run)/'checkpoints').glob('step_*.pkl')):
            if not path.with_suffix('.json').exists():continue
            step=read(path.with_suffix('.json'))['step']
            if not (root/arm/f'step_{step:05d}/summary.json').exists():result.append((step,arm,path))
    return sorted(result)


def watch(root):
    plan=read(root/'plan.json')
    for name,expected in plan['files'].items():
        if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=expected:raise ValueError('Dense code changed')
    lock=(root/'controller/watch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        active=False;history={}
        for path in sorted((root/'controller/jobs').glob('*.json')):
            row=read(path);key=row['key'];history.setdefault(key,[]).append(row)
            if row['phase']=='finished':continue
            if not row.get('job_id'):raise RuntimeError('Reconcile ambiguous submission '+str(path))
            state,code=state_of(row['job_id'])
            if state in ACTIVE or state=='ACCOUNTING_PENDING':active=True;continue
            row.update(phase='finished',slurm_state=state,exit_code=code);write(path,row)
            if (state!='COMPLETED' or code!='0:0') and (state not in RETRYABLE or len(history[key])>3):
                raise RuntimeError(f'Evaluation {key} failed: {state} {code}')
        pending=candidates(root,plan)
        if not active and pending:
            step,arm,checkpoint=pending[0];key=f'{arm}-{step:05d}';attempt=len(history.get(key,[]))+1
            args,script=batch(plan,'eval-'+key,True,[])
            command=[plan['python'],'-u',str(REPO/'experimental/residualmamba_NGCM/dense_eval.py'),
                '--root',str(root),'--arm-root',plan['arms'][arm],'--checkpoint',str(checkpoint)]
            lines=script.splitlines();lines[-1]='exec '+shlex.join(command);script='\n'.join(lines)+'\n'
            receipt=root/'controller/jobs'/f'{key}-{attempt:03d}.json'
            row={'key':key,'phase':'submitting','checkpoint':str(checkpoint),'script':script,'args':args,'submitted_unix':time.time()}
            write(receipt,row)
            out=subprocess.run(args,input=script,text=True,capture_output=True,timeout=60)
            if out.returncode:raise RuntimeError('Submission not confirmed: '+out.stderr)
            job=out.stdout.strip().split(';')[0]
            if not job.isdigit():raise RuntimeError('Unrecognized sbatch response')
            row.update(phase='submitted',job_id=job);write(receipt,row);active=True
            print(json.dumps({'submitted':key,'job_id':job}),flush=True)
        complete=not pending and not active and all((Path(run)/'report/COMPLETE.json').exists() for run in plan['arms'].values())
        write(root/'controller/status.json',{'phase':'complete' if complete else 'evaluating' if active else 'waiting_for_checkpoint',
            'heartbeat_unix':time.time(),'pending':[(step,arm) for step,arm,_ in pending]})
        if complete:return
        time.sleep(30)

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
try:watch(a.root.resolve())
except Exception as error:
    write(a.root/'controller/status.json',{'phase':'failed','error':str(error),'heartbeat_unix':time.time()});raise
