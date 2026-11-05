#!/usr/bin/env python3
"""Report download completion and observed throughput; optionally inspect local locks."""
import argparse
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT.parent/'Weather_Global/data/neuralgcm/prepared/era5_2015_2023_restore_20261104'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-local-locks',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    now=time.time()
    receipts=list(DATA.glob('shards/*/res2p8/*/frames/*.npz.json'))
    mtimes=[p.stat().st_mtime for p in receipts]
    result={'checked_unix':now,'host':socket.gethostname(),'frames':len(receipts),
        'total_frames':13148,'complete_months':len(list(DATA.glob('shards/*/res2p8/READY.json'))),
        'complete_store':(DATA/'COMPLETE.json').exists(),
        'recent_frames':{str(w):sum(now-w<t<=now for t in mtimes) for w in (60,300,600)}}
    pool=ROOT/'logs/restore_parallel_20261104/pool.json'
    if pool.exists():
        meta=json.loads(pool.read_text());started=meta['started_unix']
        result['additional_pool']=meta
        result['before_pool_600s_frames']=sum(started-600<t<started for t in mtimes)
        result['pool_age_seconds']=now-started
        if args.check_local_locks and socket.gethostname()!=meta['host']:
            parser.error('Inspect advisory locks on the staging host itself')
    rows=[]
    for name in ('restore_20261104','restore_parallel_20261104'):
        for path in sorted((ROOT/'logs'/name).glob('lane-*.json')):
            d=json.loads(path.read_text())
            row={'pool':name,'worker':d['worker'],'index':d['index'],'status':d['status']}
            if args.check_local_locks:
                lock=DATA/'download_locks'/f"month-{d['index']:03d}.lock"
                row['lock_held']=False
                if lock.exists():
                    with lock.open() as handle:
                        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:row['lock_held']=True
            rows.append(row)
    result['lanes']=rows
    if args.check_local_locks:
        processes=subprocess.check_output(['ps','-u',str(__import__('os').getuid()),'-o','args='],text=True)
        result['local_download_processes']=sum('python -u scripts/preprocessing/restore_ngcm_2p8.py month ' in line
                                                for line in processes.splitlines())
    text=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if args.output:
        if not str(args.output.resolve()).startswith('/scratch/'):parser.error('Output must be on scratch')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
