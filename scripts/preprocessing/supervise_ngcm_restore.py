#!/usr/bin/env python3
"""Bounded, restartable monthly download supervisor for the staging node."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT.parent/'Weather_Global/data/neuralgcm/prepared/era5_2015_2023_restore_20261104'
LOGS=ROOT/'logs/restore_20261104'


def main():
    lock=(LOGS/'supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (DATA/'PILOT_OK.json').is_file():
        raise RuntimeError('Verified pilot required')
    if __import__('shutil').disk_usage(DATA).free<128*1024**3:
        raise RuntimeError('Need 128 GiB available for the restored data')
    cpus=sorted(os.sched_getaffinity(0))
    pending=[i for i in range(108) if not (DATA/'shards'/f'{2015+i//12}-{1+i%12:02d}'/'res2p8/READY.json').exists()]
    def lane(worker):
        failures=[]
        affinity=','.join(map(str,cpus[4*worker:4*worker+4]))
        for index in pending[worker::4]:
            command=['taskset','-c',affinity,'bash',str(ROOT/'scripts/preprocessing/restore_ngcm_staging.sh'),'month','--index',str(index)]
            code=None
            for attempt in range(1,4):
                logfile=LOGS/f'month-{index:03d}-attempt-{attempt}-{time.time_ns()}.log'
                state={'worker':worker,'index':index,'attempt':attempt,'log':str(logfile),'status':'running'}
                temp=LOGS/f'lane-{worker}.tmp'
                temp.write_text(json.dumps(state));temp.replace(LOGS/f'lane-{worker}.json')
                with logfile.open('w') as log:
                    code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
                if code==0:break
                time.sleep(10)
            if code:failures.append(index)
        return failures
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        failures=[i for lane_failures in pool.map(lane,range(4)) for i in lane_failures]
    (LOGS/'supervisor_result.json').write_text(json.dumps({'failed_months':failures,'workers':4}))
    if failures:raise RuntimeError(f'Rerun supervisor to resume failed months: {failures}')
    subprocess.run(['bash',str(ROOT/'scripts/preprocessing/restore_ngcm_staging.sh'),'merge'],check=True)

if __name__=='__main__':main()
