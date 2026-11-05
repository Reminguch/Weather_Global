#!/usr/bin/env python3
"""Add cooperative download workers without changing the pinned data producer.

Exclude the four legacy workers' active months when attaching this pool for the
first time. Future legacy launches use the same per-month lock in the wrapper.
Only the original supervisor performs the final checksum-verified merge.
"""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT.parent/'Weather_Global/data/neuralgcm/prepared/era5_2015_2023_restore_20261104'


def finished(index):
    path=DATA/'shards'/f'{2015+index//12}-{1+index%12:02d}'/'res2p8'
    return (path/'READY.json').is_file() and (path/'profile.json').is_file()


def atomic_json(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--cpu-offset',type=int,default=16)
    p.add_argument('--exclude',type=int,nargs='*',default=[])
    p.add_argument('--log-root',type=Path,required=True)
    a=p.parse_args()
    if not str(a.log_root.resolve()).startswith('/scratch/'):
        p.error('Logs must be on scratch')
    cpus=sorted(os.sched_getaffinity(0))
    if a.workers<1 or a.cpu_offset<0 or a.cpu_offset+4*a.workers>len(cpus):
        p.error('Need four available CPU affinity slots per additional worker')
    if any(i<0 or i>=108 for i in a.exclude):p.error('Excluded month indices must be 0..107')
    if not (DATA/'PILOT_OK.json').exists():raise RuntimeError('Verified pilot required')
    if __import__('shutil').disk_usage(DATA).free<128*1024**3:
        raise RuntimeError('Need 128 GiB available for restored data')
    a.log_root.mkdir(parents=True,exist_ok=True)
    lock=(a.log_root/'supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(a.log_root/'pool.json',{'workers':a.workers,'cpu_offset':a.cpu_offset,
        'excluded_legacy_active_months':a.exclude,'host':__import__('socket').gethostname(),
        'started_unix':time.time(),'pid':os.getpid()})
    # Claim distinct months locally; the wrapper lock also coordinates with
    # independent supervisors on the same staging node. Cross-node advisory
    # locking is not assumed for this filesystem.
    guard=threading.Lock()
    claimed=set()
    failed=set()
    attempts={}
    def lane(worker):
        affinity=','.join(map(str,cpus[a.cpu_offset+4*worker:a.cpu_offset+4*worker+4]))
        deferred=set()
        while True:
            with guard:
                pending=[i for i in range(107,-1,-1) if i not in a.exclude and
                    i not in claimed and i not in failed and not finished(i)]
                choices=[i for i in pending if i not in deferred]
                if not choices:
                    if not pending:return
                    deferred.clear()
                    index=None
                else:
                    index=choices[0]
                    claimed.add(index)
            if index is None:
                time.sleep(5)
                continue
            logfile=a.log_root/f'month-{index:03d}-{time.time_ns()}.log'
            state={'worker':worker,'index':index,'log':str(logfile),'status':'running',
                   'affinity':affinity,'started_unix':time.time()}
            atomic_json(a.log_root/f'lane-{worker}.json',state)
            command=['taskset','-c',affinity,'bash',str(ROOT/'scripts/preprocessing/restore_ngcm_staging.sh'),
                     'month','--index',str(index)]
            with logfile.open('w') as out:
                code=subprocess.run(command,env=dict(os.environ,NGCM_MONTH_LOCK_NONBLOCK='1'),
                    stdout=out,stderr=subprocess.STDOUT).returncode
            with guard:
                claimed.remove(index)
                if code==75:
                    deferred.add(index)
                elif code:
                    attempts[index]=attempts.get(index,0)+1
                    if attempts[index]>=3:failed.add(index)
            state.update(status='busy' if code==75 else 'failed' if code else 'complete',
                         exit_code=code,finished_unix=time.time())
            atomic_json(a.log_root/f'lane-{worker}.json',state)
            if code and code!=75:time.sleep(10)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(lane,range(a.workers)))
    atomic_json(a.log_root/'result.json',{'failed_months':sorted(failed),'workers':a.workers,
        'finished_unix':time.time(),'merge_owner':'original supervisor'})
    if failed:raise RuntimeError(f'Months failed after retries: {sorted(failed)}')


if __name__=='__main__':main()
