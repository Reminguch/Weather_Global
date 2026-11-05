#!/usr/bin/env python3
"""Detach the authorized diagnostic scheduler with all files on scratch."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
a=p.parse_args();root=a.root.resolve()
if not str(root).startswith('/scratch/'):p.error('Use scratch')
plan=json.loads((root/'plan.json').read_text());code=Path(plan['code_snapshot'])
controller=code/'scripts/training/ngcm_2k_controller.py'
pidfile=root/'controller/watcher.pid'
if pidfile.exists():
    old=int(pidfile.read_text());cmd=Path(f'/proc/{old}/cmdline')
    if cmd.exists() and str(root).encode() in cmd.read_bytes() and b'ngcm_2k_controller.py' in cmd.read_bytes():
        raise RuntimeError(f'Watcher is already running: {old}')
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
         TMPDIR=str(code/'.tmp'),XDG_CACHE_HOME=str(code/'.cache'))
with (root/'controller/watcher.log').open('a') as log:
    process=subprocess.Popen([plan['python'],str(controller),'watch','--root',str(root)],
        cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        close_fds=True,start_new_session=True)
pidfile.write_text(str(process.pid)+'\n')
print(json.dumps({'watcher_pid':process.pid,'root':str(root)}))
