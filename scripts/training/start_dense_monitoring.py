#!/usr/bin/env python3
"""Freeze the dense monitor and safely replace controller processes only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time
REPO=Path(__file__).resolve().parents[2]
FILES=('experimental/residualmamba_NGCM/dense_eval.py','experimental/residualmamba_NGCM/dense_train.py',
       'scripts/training/dense_checkpoint_controller.py','scripts/training/watch_dense_eval.py',
       'scripts/training/start_dense_monitoring.py','scripts/training/ngcm_2k_controller.py',
       'experimental/residualmamba_NGCM/DENSE_EVAL.md')


def read(p):return json.loads(p.read_text())
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def arm(root,runs):
    if (root/'plan.json').exists():raise FileExistsError(root/'plan.json')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise RuntimeError('Commit tested source first')
    for sub in ('logs','controller/jobs'):(root/sub).mkdir(parents=True,exist_ok=True)
    pp=read(runs['tail20']/'plan.json')
    plan={'root':str(root),'python':pp['python'],'code_snapshot':str(root/'code'),
        'arms':{a:str(p) for a,p in runs.items()},'files':{n:sha(REPO/n) for n in FILES},
        'repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'scope':'monitor-only full fixed validation every available checkpoint; future continuation saves at multiples of 100'}
    subprocess.run(['git','archive','--format=tar','--output='+str(root/'source.tar'),'HEAD',
        'src','third_party/graphcast','experimental/residualmamba_NGCM','scripts/training/ngcm_2k_controller.py',
        'scripts/training/dense_checkpoint_controller.py','scripts/training/watch_dense_eval.py',
        'scripts/training/start_dense_monitoring.py'],cwd=REPO,check=True)
    with tarfile.open(root/'source.tar') as tf:tf.extractall(root/'code',filter='data')
    for name,value in plan['files'].items():
        if sha(root/'code'/name)!=value:raise ValueError('Snapshot mismatch')
    for sub in ('.cache','.tmp'):(root/'code'/sub).mkdir(exist_ok=True)
    write(root/'plan.json',plan);print(json.dumps({'armed':str(root),'commit':plan['repo_commit']}))


def detach(command,code,logpath,python):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR=str(code/'.tmp'),XDG_CACHE_HOME=str(code/'.cache'))
    with logpath.open('a') as log:
        proc=subprocess.Popen([python,*command],cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True)
    return proc.pid


def install(root):
    plan=read(root/'plan.json');code=Path(plan['code_snapshot'])
    for arm,run in plan['arms'].items():
        r=Path(run);pidfile=r/'controller/watcher.pid';old=int(pidfile.read_text())
        cmd=Path(f'/proc/{old}/cmdline')
        if cmd.exists():
            value=cmd.read_bytes()
            if str(r).encode() not in value:raise RuntimeError('PID belongs to another process')
            if b'dense_checkpoint_controller.py' in value:continue
            if not any(x in value for x in (b'ngcm_2k_controller.py',b'ngcm_tail1_controller.py')):raise RuntimeError('Unexpected controller')
            # Both workflows are running jobs already; there should be no unresolved submit.
            for p in (r/'controller/jobs').glob('*.json'):
                if read(p)['phase']=='submitting':raise RuntimeError('Submission in progress; retry installation later')
            os.kill(old,signal.SIGTERM)
            for _ in range(50):
                if not cmd.exists():break
                time.sleep(.1)
        dispatch={'code_snapshot':str(code),'files':plan['files'],'installed_unix':time.time(),
            'previous_controller_pid':old,'checkpoint_every_requested':100,
            'training_config':'unchanged; callback stop/save and exact continuation in same job'}
        write(r/'controller/dense-dispatch.json',dispatch)
        pid=detach([str(code/'scripts/training/dense_checkpoint_controller.py'),'--arm-root',str(r)],code,r/'controller/watcher.log',plan['python'])
        pidfile.write_text(str(pid)+'\n');print(json.dumps({'arm':arm,'new_controller_pid':pid,'previous_pid':old}),flush=True)
    pidfile=root/'controller/watcher.pid'
    if pidfile.exists():
        cmd=Path(f'/proc/{int(pidfile.read_text())}/cmdline')
        if cmd.exists() and str(root).encode() in cmd.read_bytes():return
    pid=detach([str(code/'scripts/training/watch_dense_eval.py'),'--root',str(root)],code,root/'controller/watcher.log',plan['python'])
    pidfile.write_text(str(pid)+'\n');print(json.dumps({'eval_controller_pid':pid}),flush=True)

p=argparse.ArgumentParser();p.add_argument('mode',choices=('arm','install'));p.add_argument('--root',type=Path,required=True)
p.add_argument('--tail20',type=Path);p.add_argument('--tail1',type=Path);a=p.parse_args();root=a.root.resolve()
if not str(root).startswith('/scratch/'):p.error('Scratch required')
if a.mode=='arm':
    if a.tail20 is None or a.tail1 is None:p.error('Both arms required')
    arm(root,{'tail20':a.tail20.resolve(),'tail1':a.tail1.resolve()})
else:install(root)
