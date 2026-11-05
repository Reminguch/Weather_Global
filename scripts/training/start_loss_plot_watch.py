#!/usr/bin/env python3
"""Detach a lightweight CPU plot refresher; no Slurm/GPU slots are requested."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
if not str(root).startswith('/scratch/'):raise ValueError('Scratch required')
plan=json.loads((root/'plan.json').read_text());out=root/'separate_plots';out.mkdir(exist_ok=True)
pidfile=out/'watcher.pid'
if pidfile.exists():
    proc=Path(f'/proc/{int(pidfile.read_text())}/cmdline')
    if proc.exists() and str(root).encode() in proc.read_bytes():raise RuntimeError('Plot watcher already running')
source=REPO/'experimental/residualmamba_NGCM/plot_checkpoint_losses.py';snapshot=root/'plotter_snapshot';snapshot.mkdir(exist_ok=True)
target=snapshot/source.name
if target.exists() and target.read_bytes()!=source.read_bytes():raise ValueError('Existing plotting snapshot differs')
shutil.copy2(source,target)
for name in ('.cache','.tmp'):(snapshot/name).mkdir(exist_ok=True)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR=str(snapshot/'.tmp'),
    XDG_CACHE_HOME=str(snapshot/'.cache'),MPLCONFIGDIR=str(snapshot/'.cache/matplotlib'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (out/'watcher.log').open('a') as log:
    proc=subprocess.Popen([plan['python'],str(target),'--root',str(root),'--output',str(out),'--watch'],cwd=snapshot,
        env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True)
pidfile.write_text(str(proc.pid)+'\n')
(out/'plotter-source.json').write_text(json.dumps({'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'script':str(target),'pid':proc.pid},indent=2)+'\n')
print(json.dumps({'plot_watcher_pid':proc.pid,'output':str(out)}))
