#!/usr/bin/env python3
"""Arm an isolated tail1 run or detach its controller on the staging host."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'experimental/residualmamba_NGCM'))
from residualmamba_ngcm.artifacts import read_json,write_json,sha256,source_identity,versions
from tail1_extension import Tail1Config

FILES=('experimental/residualmamba_NGCM/tail1_extension.py',
       'experimental/residualmamba_NGCM/tail1_2k.py',
       'experimental/residualmamba_NGCM/diagnostic_2k.py',
       'experimental/residualmamba_NGCM/TAIL1_COMPARE.md',
       'scripts/training/ngcm_tail1_controller.py','scripts/training/start_ngcm_tail1.py')
PARENT_FILES=('plan.json','setup.json','geometry.npz','geometry.json','stats.npz','stats.json',
              'cache/READY.json','cache/contract.json','qualification/qualification.json',
              'verify-cache.json','diagnostic-training.json')


def arm(root,parent):
    if (root/'plan.json').exists():raise FileExistsError('Existing run must use launch')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise RuntimeError('Commit the tested source before arming')
    root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root).free<200*1024**3:raise RuntimeError('Need 200 GiB scratch headroom')
    pp=read_json(parent/'plan.json')
    if pp['source_identity']!=source_identity() or pp['libraries']!=versions():
        raise ValueError('Core source or libraries differ from qualified parent')
    for sub in ('controller/jobs','logs','qualification'):(root/sub).mkdir(parents=True,exist_ok=True)
    plan={'root':str(root),'parent_root':str(parent),'code_snapshot':str(root/'code'),
        'python':pp['python'],'download_root':pp['download_root'],'qa_job':pp['qa_job'],
        'source_identity':source_identity(),'libraries':versions(),'config':asdict(Tail1Config()),
        'target_steps':2000,'parent_artifacts':{n:sha256(parent/n) for n in PARENT_FILES},
        'code_hashes':{n:sha256(REPO/n) for n in FILES},
        'extension_sha256':sha256(REPO/FILES[0]),
        'repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'created_unix':time.time(),'scope':'independent tail1 2k diagnostic, shared full train statistics; original tail20 continues',
        'cpu_tests_sha256':sha256(REPO/'experimental/residualmamba_NGCM/validation/20261105_tail1/cpu-tests.txt')}
    plan['orchestration_hashes']=plan['code_hashes']
    archive=root/'source.tar'
    subprocess.run(['git','archive','--format=tar','--output='+str(archive),'HEAD',
        'AGENTS.md','src','third_party/graphcast','experimental/residualmamba_NGCM',
        'scripts/training/ngcm_tail1_controller.py','scripts/training/start_ngcm_tail1.py',
        'scripts/training/ngcm_2k_controller.py'],cwd=REPO,check=True)
    with tarfile.open(archive) as tf:tf.extractall(root/'code',filter='data')
    for name,value in plan['code_hashes'].items():
        if sha256(root/'code'/name)!=value:raise ValueError('Snapshot mismatch')
    for sub in ('.tmp','.cache'):(root/'code'/sub).mkdir(exist_ok=True)
    shutil.copy2(parent/'qualification/qualification.json',root/'qualification/qualification.json')
    write_json(root/'plan.json',plan,immutable=True)
    write_json(root/'controller/status.json',{'phase':'armed','step':0,'created_unix':time.time()})
    print(json.dumps({'armed':str(root),'commit':plan['repo_commit']}))


def launch(root):
    plan=read_json(root/'plan.json');code=Path(plan['code_snapshot']);pidfile=root/'controller/watcher.pid'
    if pidfile.exists():
        cmd=Path(f'/proc/{int(pidfile.read_text())}/cmdline')
        if cmd.exists() and str(root).encode() in cmd.read_bytes():raise RuntimeError('Controller already running')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR=str(code/'.tmp'),XDG_CACHE_HOME=str(code/'.cache'))
    with (root/'controller/watcher.log').open('a') as log:
        proc=subprocess.Popen([plan['python'],str(code/'scripts/training/ngcm_tail1_controller.py'),'watch','--root',str(root)],
            cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True)
    pidfile.write_text(str(proc.pid)+'\n');print(json.dumps({'watcher_pid':proc.pid,'root':str(root)}))

p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('arm','launch'))
p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path)
a=p.parse_args();root=a.root.resolve()
if not str(root).startswith('/scratch/'):p.error('Scratch required')
if a.mode=='arm':
    if a.parent is None:p.error('--parent required')
    arm(root,a.parent.resolve())
else:launch(root)
