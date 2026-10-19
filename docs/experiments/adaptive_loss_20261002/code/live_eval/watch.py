"""Bounded CPU collector and scheduler for live checkpoint evaluation."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu'
from common import read,write
from capture import capture

def submit_pending(root):
    ledger=read(root/'jobs.json') if (root/'jobs.json').exists() else {}
    ids=[x['job_id'] for x in ledger.values()]
    states={}
    if ids:
        result=subprocess.run(['sacct','-X','-P','-n','-j',','.join(ids),'--format=JobID,State'],check=True,text=True,capture_output=True)
        for row in result.stdout.splitlines():
            parts=row.split('|')
            if len(parts)>1:states[parts[0]]=parts[1].split()[0]
    assigned={}
    for name,entry in ledger.items():
        entry['observed_state']=states.get(entry['job_id'],'UNKNOWN')
        for identity in entry['snapshots']:assigned.setdefault(identity,[]).append(entry)
    candidates=[]
    for path in (root/'snapshots').glob('*/metadata.json'):
        meta=read(path);identity=meta['id']
        if (root/'results'/(identity+'.json')).exists():continue
        previous=assigned.get(identity,[])
        terminal={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','COMPLETED'}
        if previous and (len(previous)>=2 or any(x['observed_state'] not in terminal for x in previous)):continue
        candidates.append(meta)
    candidates.sort(key=lambda x:(x['kind']!='last',-x['update'],x['hardware'],x['mode']))
    active=[x['job_id'] for x in ledger.values() if x['observed_state'] not in {'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','COMPLETED'}]
    previous=active[-1] if active else None
    next_number=max([int(k.split('_')[-1]) for k in ledger] or [0])+1
    for offset in range(0,len(candidates),3):
        snapshots=[x['id'] for x in candidates[offset:offset+3]];name='batch_'+str(next_number).zfill(4);next_number+=1
        path=root/'batches'/(name+'.json');write(path,dict(snapshots=snapshots))
        cmd=['sbatch','--parsable','--job-name=ngcm-live-'+name]
        if previous:cmd+=['--dependency=afterany:'+previous]
        cmd += [str(root/'run_eval.sbatch'),str(Path(__file__).with_name('evaluate.py')),'--root',str(root),'--batch',str(path)]
        result=subprocess.run(cmd,check=True,text=True,capture_output=True);job=result.stdout.strip().split(';')[0]
        if not job.isdecimal():raise RuntimeError('Unrecognized Slurm response')
        ledger[name]=dict(job_id=job,snapshots=snapshots,observed_state='SUBMITTED');write(root/'jobs.json',ledger)
        print(json.dumps(dict(submitted=job,snapshots=snapshots)),flush=True);previous=job
    write(root/'jobs.json',ledger)
    return ledger

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--max-seconds',type=int,default=21000);p.add_argument('--once',action='store_true');a=p.parse_args()
    lock=(a.root/'.watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(a.root/'plan.json');start=time.monotonic()
    while True:
        capture(a.root);ledger=submit_pending(a.root)
        subprocess.run([sys.executable,str(Path(__file__).with_name('report.py')),'--root',str(a.root)],check=True)
        terminal=[]
        for folder in plan['campaigns'].values():
            for mode in ('fixed','calibrated','adaptive'):
                run=Path(folder)/('trial_'+mode);terminal.append((run/'DONE.json').exists() or (run/'failure.json').exists())
        pending=[p.parent.name for p in (a.root/'snapshots').glob('*/metadata.json') if not (a.root/'results'/(p.parent.name+'.json')).exists()]
        status=dict(checked_at=time.time(),all_training_terminal=all(terminal),unevaluated_snapshots=pending,watcher_job_id=os.environ.get('SLURM_JOB_ID'))
        write(a.root/'watch_status.json',status)
        if a.once or (all(terminal) and not pending):return
        if time.monotonic()-start>a.max_seconds:
            status['watcher_time_budget_reached']=True;write(a.root/'watch_status.json',status);return
        time.sleep(60)

if __name__=='__main__':main()
