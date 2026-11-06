#!/usr/bin/env python3
"""Run all matched K1 checkpoint evaluations on up to three ailab H200s."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/training'))
from ngcm_2k_controller import read,write,sha,state_of,ACTIVE,RETRYABLE


def priority_steps(root):
    steps=sorted(read(p)['step'] for p in (root/'checkpoints').glob('step_*.json') if p.with_suffix('.pkl').exists())
    if not steps or steps[-1]!=2000:
        raise ValueError('Requires a completed 2000-update run')
    return [2000]+[s for s in steps if s!=2000]


def submit(root,out,step,attempt):
    template=read(root/'controller/jobs/validate-dense-027.json')
    plan=read(root/'plan.json');worker=out/'code/matched_v2_eval.py'
    command=[plan['python'],'-u',str(worker),'--root',str(root),'--output',str(out/'tail1'),'--step',str(step)]
    script=template['script'].split('exec ')[0]+'exec '+shlex.join(command)+'\n'
    args=[x for x in template['sbatch'] if not x.startswith(('--job-name=','--output='))]
    args+=['--job-name=ngcm-v2-matched-k1-'+str(step),'--output='+str(out/'logs'/f'step-{step}-%j.log')]
    path=out/'jobs'/f'{step:05d}-{attempt:02d}.json'
    record={'step':step,'attempt':attempt,'phase':'submitting','script':script,'sbatch':args,'submitted_unix':time.time()}
    write(path,record)
    result=subprocess.run(args,input=script,text=True,capture_output=True,timeout=60)
    if result.returncode:
        record.update(phase='rejected',reason=result.stderr);write(path,record)
        raise RuntimeError(result.stderr)
    job=result.stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Ambiguous submission: '+result.stdout)
    record.update(phase='submitted',job_id=job);write(path,record)
    print(f'submitted matched K1 step={step} job={job}',flush=True)


def watch(root,out,report):
    lock=(out/'watch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(out/'plan.json');steps=priority_steps(root)
    report_script=REPO/'experimental/residualmamba_NGCM/report_matched_v2.py'
    python=read(root/'plan.json')['python'];last_count=-1
    while True:
        if (out/'STOP').exists():raise RuntimeError('Explicit stop marker')
        if sha(out/'code/matched_v2_eval.py')!=plan['worker_sha256']:
            raise RuntimeError('Matched worker changed')
        active={};history={}
        for path in sorted((out/'jobs').glob('*.json')):
            row=read(path);step=row['step'];history.setdefault(step,[]).append(row)
            if row['phase']=='finished':continue
            if not row.get('job_id'):raise RuntimeError('Unconfirmed submission; reconcile '+str(path))
            state,code=state_of(row['job_id'])
            if state in ACTIVE or state=='ACCOUNTING_PENDING':active[step]=row['job_id'];continue
            row.update(phase='finished',slurm_state=state,exit_code=code);write(path,row)
            if (state!='COMPLETED' or code!='0:0') and (state not in RETRYABLE or row['attempt']>=3):
                raise RuntimeError(f'Job {row["job_id"]}: {state} {code}')
        complete=[s for s in steps if (out/'tail1'/f'step_{s:05d}/summary.json').exists()]
        if len(complete)!=last_count and report_script.exists():
            subprocess.run([python,str(report_script),'--matched-root',str(out),'--output',str(report)],check=True)
            last_count=len(complete)
        if len(complete)==len(steps) and not active:
            write(out/'COMPLETE.json',{'checkpoints':len(steps),'steps':steps,'training_k':1,'evaluation_k':1})
            write(out/'status.json',{'phase':'complete','completed':len(steps),'total':len(steps),'pid':os.getpid()})
            # Publish only the generated report directory, never arbitrary worktree changes.
            staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
            if staged:raise RuntimeError('Evaluation complete; unrelated staged changes prevent automatic publication')
            subprocess.run(['git','add','--',str(report)],cwd=REPO,check=True)
            changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO).returncode
            if changed:
                subprocess.run(['git','commit','-m','Publish complete training-matched K1 and K20 validation curves'],cwd=REPO,check=True)
                subprocess.run(['git','push','origin','residualmamba_NGCM_rebuild_20261104'],cwd=REPO,check=True)
            write(out/'PUBLISHED.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()})
            return
        # Gate parallel full evaluation on the real H200 first-batch smoke.
        qualified=(out/'tail1/step_02000/smoke.json').exists()
        capacity=3 if qualified else 1
        for step in steps:
            if step in complete or step in active:continue
            if len(active)>=capacity:break
            submit(root,out,step,len(history.get(step,[]))+1);active[step]='submitted'
        write(out/'status.json',{'phase':'running','completed':len(complete),'total':len(steps),
            'active':active,'gpu_smoke_passed':qualified,'heartbeat_unix':time.time(),'pid':os.getpid()})
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','report'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:watch(a.root.resolve(),a.output.resolve(),a.report.resolve())
    except Exception as error:
        write(a.output/'status.json',{'phase':'failed','error':str(error),'pid':os.getpid()});raise
