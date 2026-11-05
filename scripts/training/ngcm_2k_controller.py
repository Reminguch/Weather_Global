#!/usr/bin/env python3
"""Durable download -> preparation -> H200 gpu-test 2k diagnostic controller."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

REPO=Path(__file__).resolve().parents[2]
WORKER=REPO/'experimental/residualmamba_NGCM/diagnostic_2k.py'
ACTIVE={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','REQUEUED'}
RETRYABLE={'TIMEOUT','PREEMPTED','NODE_FAIL','BOOT_FAIL'}


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temporary.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_snapshot(plan):
    code=Path(plan.get('code_snapshot',REPO))
    for name,expected in plan['orchestration_hashes'].items():
        if sha(code/name)!=expected:raise RuntimeError('Controller/worker source changed: '+name)


def latest_step(root):
    steps=[]
    for receipt in (Path(root)/'checkpoints').glob('step_*.json'):
        value=read(receipt);p=receipt.with_suffix('.pkl')
        if not p.exists():continue
        steps.append(int(value['step']))
    return max(steps,default=0)


def next_actions(root):
    """Return desired logical jobs; artifact completion drives restartable stages."""
    root=Path(root)
    if not (root/'setup.json').exists():return [('setup',False,[])]
    actions=[]
    if not (root/'stats.json').exists():actions.append(('stats',False,[]))
    for lane in range(3):
        if not (root/f'cache-lane-{lane}.json').exists():actions.append((f'cache-{lane}',True,['--lane',str(lane)]))
    if actions:return actions
    if not (root/'cache/READY.json').exists():return [('cache-finalize',True,[])]
    if not (root/'diagnostic-training.json').exists():return [('checks',True,[])]
    step=latest_step(root)
    for endpoint in (1000,2000):
        if step<endpoint:return [(f'train-{endpoint}',True,['--endpoint',str(endpoint)])]
        if not (root/f'checkpoints/validation_{endpoint:05d}.json').exists():
            return [(f'validate-{endpoint}',True,['--endpoint',str(endpoint)])]
    if not (root/'report/COMPLETE.json').exists():return [('report',False,[])]
    return []


def desired_actions(root,data_ready,qualification_ready,qa_ready):
    actions=[]
    if not qualification_ready and qa_ready:actions.append(('qualify',False,[]))
    if data_ready:
        preparation=next_actions(root)
        if qualification_ready or (preparation and preparation[0][0] not in
                ('checks','train-1000','validate-1000','train-2000','validate-2000','report')):
            actions+=preparation
    return actions


def batch(plan,key,gpu,extra):
    root=Path(plan['root'])
    code=Path(plan.get('code_snapshot',REPO))
    worker=code/WORKER.relative_to(REPO)
    command=[plan['python'],'-u',str(worker),key.split('-')[0] if key.startswith(('train-','validate-')) else key,
             '--root',str(root),*extra]
    if key.startswith('cache-') and key!='cache-finalize':command[3]='cache'
    env={'TMPDIR':str(code/'.tmp'),'XDG_CACHE_HOME':str(code/'.cache'),
         'MPLCONFIGDIR':str(code/'.cache/matplotlib'),'PYTHONPATH':str(code/'experimental/residualmamba_NGCM'),
         'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1',
         'JAX_PLATFORMS':'cuda' if gpu else 'cpu','XLA_PYTHON_CLIENT_PREALLOCATE':'false',
         'XLA_FLAGS':'--xla_gpu_enable_scatter_determinism_expander=true --xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0',
         'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
         'NGCM_GPU_EXECUTION_POLICY':'deterministic_gpu_v1'}
    script='#!/usr/bin/env bash\nset -euo pipefail\nunset LD_LIBRARY_PATH\n'
    script+='cd '+shlex.quote(str(code))+'\n'
    script+=''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())
    script+='exec '+shlex.join(command)+'\n'
    args=['sbatch','--parsable','--account=mengdiw','--cpus-per-task=8',
          '--job-name=ngcm2k-'+key,'--output='+str(root/'logs'/f'{key}-%j.log')]
    if gpu:args+=['--partition=ailab','--qos=gpu-test','--gres=gpu:h200:1','--mem=96G','--time=01:00:00']
    else:args+=['--partition=cpu','--qos=short','--mem=48G','--time=04:00:00' if key=='stats' else '--time=00:30:00']
    return args,script


def state_of(job):
    query=subprocess.run(['squeue','-h','-j',job,'-o','%T'],capture_output=True,text=True,timeout=30)
    if query.returncode and 'Invalid job id' not in query.stderr:
        raise RuntimeError('Unable to query scheduler: '+query.stderr)
    text=query.stdout.strip()
    if text:return text.splitlines()[0],None
    text=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'-o','JobIDRaw,State,ExitCode'],text=True,timeout=30)
    for line in text.splitlines():
        row=line.split('|')
        if row[0]==job:return row[1].split()[0].rstrip('+'),row[2]
    return 'ACCOUNTING_PENDING',None


def submit(plan,key,gpu,extra,attempt):
    root=Path(plan['root']);args,script=batch(plan,key,gpu,extra)
    receipt=root/'controller/jobs'/f'{key}-{attempt:03d}.json'
    if receipt.exists():raise RuntimeError('Submission receipt already exists: '+str(receipt))
    row={'key':key,'attempt':attempt,'gpu':gpu,'sbatch':args,'script':script,
         'submitted_unix':time.time(),'phase':'submitting','step_before':latest_step(root)}
    write(receipt,row)
    result=subprocess.run(args,input=script,capture_output=True,text=True,timeout=60)
    if result.returncode:
        if 'MaxJobCount limit reached' in result.stderr or 'QOSMaxSubmitJobPerUserLimit' in result.stderr:
            row.update(phase='rejected',reason=result.stderr,finished_unix=time.time())
            write(receipt,row);return
        raise RuntimeError('Submission not confirmed; reconcile receipt before retrying: '+result.stderr)
    job=result.stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Unrecognized sbatch response: '+result.stdout)
    row.update(job_id=job,phase='submitted');write(receipt,row)
    print(json.dumps({'submitted':key,'job_id':job,'attempt':attempt}),flush=True)


def watch(root):
    root=Path(root);plan=read(root/'plan.json');assert_snapshot(plan)
    lock=(root/'controller/watch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    failures={}
    while True:
        assert_snapshot(plan)
        status={'host':socket.gethostname(),'pid':os.getpid(),'heartbeat_unix':time.time()}
        qualification_ready=(root/'qualification/qualification.json').exists()
        qa_ready=qualification_ready
        if not qualification_ready:
            qa_state,qa_exit=state_of(str(plan['qa_job']))
            if qa_state not in ACTIVE and qa_state!='ACCOUNTING_PENDING':
                if qa_state!='COMPLETED' or qa_exit!='0:0':raise RuntimeError(f'GPU regression ended {qa_state} {qa_exit}')
                qa_ready=True
        marker=Path(plan['download_root'])/'COMPLETE.json'
        data_ready=marker.exists()
        if data_ready:
            complete=read(marker)
            if complete.get('frames')!=13148:raise RuntimeError('Incomplete data marker')
        running=set();history={}
        for path in sorted((root/'controller/jobs').glob('*.json')):
            row=read(path);key=row['key'];history.setdefault(key,[]).append(row)
            if row['phase']=='rejected':continue
            if not row.get('job_id'):raise RuntimeError('Ambiguous submission requires reconciliation: '+str(path))
            if row['phase']=='finished':continue
            state,code=state_of(row['job_id'])
            if state in ACTIVE or state=='ACCOUNTING_PENDING':running.add(key);continue
            row.update(phase='finished',slurm_state=state,exit_code=code,finished_unix=time.time());write(path,row)
            if state!='COMPLETED' or code!='0:0':
                failures[key]=failures.get(key,0)+1
                if state not in RETRYABLE or failures[key]>3:
                    raise RuntimeError(f'{key} job {row["job_id"]} ended {state} {code}; see its log')
        actions=desired_actions(root,data_ready,qualification_ready,qa_ready)
        if (root/'report/COMPLETE.json').exists() and not running:
            write(root/'controller/status.json',{**status,'phase':'complete','step':latest_step(root)});return
        for key,gpu,extra in actions:
            if key in running:continue
            prior=history.get(key,[])
            if prior and prior[-1]['phase']=='rejected' and time.time()-prior[-1]['finished_unix']<60:continue
            if prior and prior[-1].get('slurm_state')=='COMPLETED' and key.startswith('train-'):
                if latest_step(root)<=prior[-1]['step_before']:raise RuntimeError('Successful training invocation made no checkpoint progress')
            submit(plan,key,gpu,extra,len(prior)+1)
        phase='running' if actions or running else 'waiting_for_verified_download' if not data_ready else 'waiting_for_gpu_restart_regression'
        write(root/'controller/status.json',{**status,'phase':phase,'step':latest_step(root),
             'download_ready':data_ready,'qualification_ready':qualification_ready,
             'qa_job':plan['qa_job'],'logical_jobs':[a[0] for a in actions]})
        time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('arm','watch','status'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--qa-job',type=int)
    a=p.parse_args();root=a.root.resolve()
    if not str(root).startswith('/scratch/'):p.error('Use scratch')
    if a.mode=='status':print(json.dumps(read(root/'controller/status.json'),indent=2));return
    if a.mode=='arm':
        if not a.qa_job:p.error('--qa-job is required')
        if (root/'plan.json').exists():raise FileExistsError('Existing run must resume with watch')
        root.mkdir(parents=True,exist_ok=True)
        for path in ('logs','controller/jobs'): (root/path).mkdir(parents=True,exist_ok=True)
        python=str(REPO.parent/'Weather_Global/.venv_ngcm_data/bin/python')
        ident=json.loads(subprocess.check_output([python,str(WORKER),'identity','--root',str(root)],text=True))
        quota=subprocess.check_output(['checkquota'],text=True,timeout=30)
        project=[line for line in quota.splitlines() if 'MENGDIW' in line]
        if not project:raise RuntimeError('Project quota audit not available')
        plan={**ident,'root':str(root),'python':python,'target_steps':2000,
              'code_snapshot':str(root/'code'),
              'repo_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
              'download_root':str(REPO.parent/'Weather_Global/data/neuralgcm/prepared/era5_2015_2023_restore_20261104'),
              'checkpoint':str(REPO.parent/'Weather_Global/data/neuralgcm/checkpoints/deterministic_2_8_deg.pkl'),
              'gc_checkpoint':str(REPO/'data/prerequisites/graphcast_small.npz'),
              'qa_job':a.qa_job,'qa_root':str(REPO/'logs'/f'deterministic-resume-{a.qa_job}'),
              'quota_audit':{'checked_unix':time.time(),'command':'checkquota','project_lines':project},
              'orchestration_hashes':{str(path.relative_to(REPO)):sha(path) for path in
                  (Path(__file__),WORKER,REPO/'experimental/residualmamba_NGCM/DIAGNOSTIC_2K.md')},
              'scope':'user-requested 2000-update loss diagnostic; annual baseline audit and full production claim deferred'}
        write(root/'plan.json',plan)
        write(root/'controller/status.json',{'phase':'armed','created_unix':time.time()})
        print(json.dumps({'armed':str(root),'target_steps':2000}));return
    try:watch(root)
    except Exception as error:
        write(root/'controller/status.json',{'phase':'failed','error':str(error),'heartbeat_unix':time.time()})
        raise


if __name__=='__main__':main()
