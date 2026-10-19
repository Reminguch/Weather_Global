#!/usr/bin/env python3
"""Plan, submit, resume and merge 2015–2023 ERA5 monthly dual-grid downloads."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.neuralgcm_residual.io import digest, read_json, sha256, write_json, versions


def prepare(root, checkpoint2, checkpoint1):
    root = Path(root).resolve()
    months = []
    for year in range(2015, 2024):
        for month in range(1, 13):
            next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
            months.append({"id": f"{year}-{month:02d}", "start": f"{year}-{month:02d}-01T00",
                           "end": f"{next_year}-{next_month:02d}-01T00"})
    files = {}
    for directory in ("src/models/neuralgcm_residual", "scripts/preprocessing"):
        for source in sorted((ROOT / directory).glob("*.py")):
            relative = source.relative_to(ROOT)
            target = root / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha256(target) != sha256(source):
                raise FileExistsError("Downloader source already frozen; use a new output root")
            shutil.copyfile(source, target)
            files[str(relative)] = sha256(target)
    manifest = {"format": "neuralgcm_era5_download_v1", "source_id": digest(files), "source_files": files,
                "source_root": str(root / "source"), "python": str(Path(sys.executable).absolute()),
                "data_source": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
                "libraries": versions(), "months": months, "cadence_hours": 6,
                "split": {"train": list(range(2015, 2022)), "val": [2022], "test": [2023]},
                "checkpoints": {"res2p8": {"path": str(Path(checkpoint2).resolve()), "sha256": sha256(checkpoint2)},
                                "res1p4": {"path": str(Path(checkpoint1).resolve()), "sha256": sha256(checkpoint1)}}}
    write_json(root / "download_manifest.json", manifest, immutable=True)
    (root / "logs").mkdir(exist_ok=True)
    return manifest


def load(root):
    root = Path(root).resolve()
    manifest = read_json(root / "download_manifest.json")
    for rel, expected in manifest["source_files"].items():
        if sha256(Path(manifest["source_root"]) / rel) != expected:
            raise ValueError(f"Downloader source changed: {rel}")
    return manifest


def worker(root, index, *, probe=False):
    manifest = load(root)
    if versions() != manifest["libraries"]:
        raise ValueError("Downloader dependency versions changed")
    for checkpoint in manifest["checkpoints"].values():
        if sha256(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("Checkpoint changed")
    import xarray as xr
    if probe:
        ds = xr.open_zarr(manifest["data_source"], chunks=None, storage_options={"token": "anon"})
        sample = ds.temperature.sel(time="2015-01-01T00", level=850).isel(latitude=slice(0,2),longitude=slice(0,2)).values
        write_json(Path(root) / "network_probe.json", {"passed": True, "shape": list(sample.shape),
                   "source_id": manifest["source_id"]}, immutable=True)
        print('ARCO-ERA5 compute-node network probe passed',flush=True)
        return
    from src.models.neuralgcm_residual.backbone import FrozenBackbone
    from src.models.neuralgcm_residual.data import prepare_era5_pair
    month = manifest["months"][index]
    out = Path(root) / "shards" / month["id"]
    if (out / "COMPLETE.json").exists():
        print('Already complete:',month['id'],flush=True)
        return
    models = {rid: FrozenBackbone.load(cp["path"], cp["sha256"]).model for rid, cp in manifest["checkpoints"].items()}
    report = prepare_era5_pair(manifest["data_source"], models, {rid: out/rid for rid in models},month["start"],month["end"])
    write_json(out / "profile.json", report)
    write_json(out / "COMPLETE.json", {"month": month["id"], "source_id": manifest["source_id"],
               "ready": {rid: sha256(out / rid / "READY.json") for rid in models}}, immutable=True)


def merge(root):
    manifest = load(root)
    from src.models.neuralgcm_residual.data import merge_prepared, PreparedStore
    import numpy as np
    for month in manifest["months"]:
        completed = read_json(Path(root) / "shards" / month["id"] / "COMPLETE.json")
        if completed["source_id"] != manifest["source_id"]:
            raise ValueError("Mixed source identities in monthly data")
    expected = np.arange(np.datetime64('2015-01-01T00'),np.datetime64('2024-01-01T00'),np.timedelta64(6,'h'))
    for rid in ('res2p8','res1p4'):
        shards = [Path(root) / 'shards' / m['id'] / rid for m in manifest['months']]
        merge_prepared(shards,Path(root)/rid)
        store = PreparedStore(Path(root)/rid)
        np.testing.assert_array_equal(store.times,expected)
    write_json(Path(root)/'READY.json',{'source_id':manifest['source_id'],'timestamps_per_resolution':len(expected),
               'months':108,'split':manifest['split']},immutable=True)


def submit(root, *, probe=False, dry_run=False, resume=False, cpus=4, memory_gb=16, hours=4):
    root=Path(root).resolve()
    manifest=load(root)
    if not probe:
        checked=read_json(root/'network_probe.json')
        if not checked['passed'] or checked['source_id']!=manifest['source_id']:
            raise ValueError('A successful compute-node network probe is required first')
    indices=[i for i,m in enumerate(manifest['months']) if not (resume and (root/'shards'/m['id']/'COMPLETE.json').exists())]
    if not indices: return {'complete':True}
    indices=[0] if probe else indices
    name='ngcm-era5-probe' if probe else 'ngcm-era5-2015-2023'
    args=['sbatch','--parsable','--job-name',name,'--cpus-per-task',str(cpus),'--mem',f'{memory_gb}G',
          '--time',f'{hours}:00:00','--array',','.join(map(str,indices)),
          '--output',str(root/'logs'/(name+'-%A_%a.out')),'--error',str(root/'logs'/(name+'-%A_%a.err'))]
    cmd=[manifest['python'],'-u',str(Path(manifest['source_root'])/'scripts/preprocessing/download_neuralgcm_era5.py'),
         'worker','--root',str(root)]
    if probe: cmd.append('--probe')
    script='#!/usr/bin/env bash\nset -euo pipefail\nexport JAX_PLATFORMS=cpu\nexport PYTHONDONTWRITEBYTECODE=1\n'
    script+='cd '+shlex.quote(manifest['source_root'])+'\nexec '+shlex.join(cmd)+'\n'
    result={'sbatch':args,'script':script,'monthly_tasks':len(indices),'array_concurrency_throttle':None}
    if not dry_run:
        submitted=subprocess.run(args,input=script,text=True,capture_output=True,check=True)
        job_id=submitted.stdout.strip().split(';')[0]
        if not job_id.isdigit(): raise ValueError(submitted.stdout)
        result['job_id']=job_id
        write_json(root/('probe_submission.json' if probe else f'download_submission_{job_id}.json'),result,immutable=True)
        if not probe:
            merge_cmd=[manifest['python'],cmd[2],'merge','--root',str(root)]
            merge_args=['sbatch','--parsable','--job-name','ngcm-era5-merge','--cpus-per-task','1','--mem','4G',
                        '--time','01:00:00','--dependency','afterok:'+job_id,'--output',str(root/'logs/merge-%j.out')]
            merge_script='#!/usr/bin/env bash\nset -euo pipefail\nexport JAX_PLATFORMS=cpu\nexec '+shlex.join(merge_cmd)+'\n'
            followup=subprocess.run(merge_args,input=merge_script,text=True,capture_output=True,check=True)
            result['merge_job_id']=followup.stdout.strip().split(';')[0]
            write_json(root/f'merge_submission_{job_id}.json',{'afterok':job_id,'job_id':result['merge_job_id']},immutable=True)
    return result


def supervise(root, *, workers=2, threads_per_worker=4, worker_python=None):
    """Durable internet-enabled staging-node alternative to offline Slurm nodes.

    Month workers execute the already-frozen source. This limits CPU/network
    use of a shared staging host, independently of the unthrottled GPU matrix.
    """
    import concurrent.futures
    import fcntl
    import time
    root=Path(root).resolve()
    manifest=load(root)
    worker_python=str(Path(worker_python or manifest['python']).absolute())
    if not Path(worker_python).is_file(): raise FileNotFoundError(worker_python)
    lock=(root/'supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if workers<1 or threads_per_worker<1: raise ValueError('Positive worker/thread counts required')
    cpus=sorted(os.sched_getaffinity(0))
    if workers*threads_per_worker>len(cpus): raise ValueError('Requested CPU affinity exceeds available CPUs')
    pending=[i for i,m in enumerate(manifest['months']) if not (root/'shards'/m['id']/'COMPLETE.json').exists()]
    def process_lane(lane):
        results=[]
        assigned=cpus[lane*threads_per_worker:(lane+1)*threads_per_worker]
        for index in pending[lane::workers]:
            month=manifest['months'][index]['id']
            command=['taskset','-c',','.join(map(str,assigned)),worker_python,'-u',
                     str(Path(manifest['source_root'])/'scripts/preprocessing/download_neuralgcm_era5.py'),
                     'worker','--root',str(root),'--index',str(index)]
            env=dict(os.environ,JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1',
                     OMP_NUM_THREADS=str(threads_per_worker),OPENBLAS_NUM_THREADS=str(threads_per_worker),
                     MKL_NUM_THREADS=str(threads_per_worker))
            code=None
            for attempt in range(3):
                log=root/'logs'/f'{month}-attempt{attempt+1}-{time.time_ns()}.log'
                write_json(root/f'lane_{lane}.json',{'month':month,'index':index,'attempt':attempt+1,'log':str(log),
                           'command':command,'status':'running'})
                with log.open('w') as f:
                    code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
                if code==0:break
                time.sleep(10)
            results.append({'month':month,'returncode':code})
            write_json(root/f'lane_{lane}.json',{'status':'progress','results':results})
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results=[entry for lane in pool.map(process_lane,range(workers)) for entry in lane]
    write_json(root/'supervisor_result.json',{'results':results,'workers':workers,'threads_per_worker':threads_per_worker})
    if any(r['returncode'] for r in results): raise RuntimeError('Some downloads failed; rerun supervise to resume')
    merge(root)
    return {'completed':True,'months':108}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('prepare','worker','submit','merge','status','supervise'):
        q=sub.add_parser(name);q.add_argument('--root',type=Path,required=True)
        if name=='prepare':
            q.add_argument('--checkpoint-2p8',type=Path,required=True);q.add_argument('--checkpoint-1p4',type=Path,required=True)
        if name in ('submit','worker'):q.add_argument('--probe',action='store_true')
        if name=='submit':
            q.add_argument('--dry-run',action='store_true');q.add_argument('--resume',action='store_true')
            q.add_argument('--cpus',type=int,default=4);q.add_argument('--memory-gb',type=int,default=16);q.add_argument('--hours',type=int,default=4)
        if name=='worker':q.add_argument('--index',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))
        if name=='supervise':
            q.add_argument('--workers',type=int,default=2);q.add_argument('--threads-per-worker',type=int,default=4)
            q.add_argument('--worker-python',type=Path,help='Equivalent node-local environment; workers still verify pinned dependency versions')
    a=p.parse_args()
    if a.command=='prepare':result=prepare(a.root,a.checkpoint_2p8,a.checkpoint_1p4)
    elif a.command=='worker':result=worker(a.root,a.index,probe=a.probe)
    elif a.command=='merge':result=merge(a.root)
    elif a.command=='submit':result=submit(a.root,probe=a.probe,dry_run=a.dry_run,resume=a.resume,cpus=a.cpus,memory_gb=a.memory_gb,hours=a.hours)
    elif a.command=='supervise':result=supervise(a.root,workers=a.workers,threads_per_worker=a.threads_per_worker,worker_python=a.worker_python)
    else:
        m=load(a.root)
        done=[month['id'] for month in m['months'] if (a.root/'shards'/month['id']/'COMPLETE.json').exists()]
        frames={rid:len(list((a.root/'shards').glob(f'*/{rid}/*/frames/*.npz.json'))) for rid in ('res2p8','res1p4')}
        lanes=[]
        for lane in sorted(a.root.glob('lane_*.json')):
            record=read_json(lane)
            lanes.append({k:record[k] for k in ('month','status','attempt','log') if k in record})
        result={'completed_months':len(done),'required_months':108,'complete':(a.root/'READY.json').exists(),
                'prepared_frames':frames,'expected_frames_per_resolution':13148,'lanes':lanes,'months':done}
    import json
    print(json.dumps(result,indent=2,default=str))


if __name__=='__main__':main()
