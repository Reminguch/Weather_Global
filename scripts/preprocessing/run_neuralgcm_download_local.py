#!/usr/bin/env python3
"""CPU-only, node-local runtime for the unchanged frozen ERA5 producer.

Each lane imports once and reuses selected model geometries for assigned months.
Dataset source identity and dependency/checkpoint hashes stay unchanged. All data
and receipts go to the specified scratch root; disposable runtime lives locally.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time
import types

RUNTIME = Path(__file__).resolve().parent


def checked_manifest(root, resolutions=('res2p8',)):
    m = json.loads((root / 'download_manifest.json').read_text())
    for rel, expected in m['source_files'].items():
        if hashlib.sha256((RUNTIME / 'source' / rel).read_bytes()).hexdigest() != expected:
            raise ValueError('Runtime source changed: ' + rel)
    m['checkpoints'] = {rid:m['checkpoints'][rid] for rid in resolutions}
    for rid, cp in m['checkpoints'].items():
        path = RUNTIME / 'checkpoints' / Path(cp['path']).name
        if hashlib.sha256(path.read_bytes()).hexdigest() != cp['sha256']:
            raise ValueError('Runtime checkpoint changed: ' + rid)
        cp['path'] = str(path)
    for name, version in m['libraries'].items():
        actual = importlib.metadata.version(name)
        if actual != version:
            raise ValueError(f'{name}: pinned {version}, found {actual}')
    # Isolate the frozen namespace from any mutable checkout on PYTHONPATH.
    for name, relative in [('src', 'src'), ('src.models', 'src/models')]:
        module = types.ModuleType(name)
        module.__path__ = [str(RUNTIME / 'source' / relative)]
        sys.modules[name] = module
    return m


def event(message, **fields):
    print(json.dumps({'at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'event': message, **fields}), flush=True)


def benchmark(root, output, frames, resolutions):
    started = time.monotonic()
    m = checked_manifest(root, resolutions)
    from src.models.neuralgcm_residual.backbone import FrozenBackbone
    from src.models.neuralgcm_residual.data import prepare_era5_pair
    import numpy as np
    models = {rid: FrozenBackbone.load(cp['path'], cp['sha256']).model for rid, cp in m['checkpoints'].items()}
    event('benchmark_models_ready', seconds=time.monotonic()-started)
    start = np.datetime64('2020-01-01T00', 'h')
    report = prepare_era5_pair(m['data_source'], models, {rid:output/rid for rid in models},
                               str(start), str(start+np.timedelta64(6*frames,'h')))
    event('benchmark_complete', seconds=time.monotonic()-started,
          affinity=sorted(os.sched_getaffinity(0)), timings=report['new_frame_timings'])


def month_complete(root, month, resolutions):
    return all((root/'shards'/month/rid/'READY.json').is_file() for rid in resolutions)


def lane(root, indices, lane_id, resolutions):
    started = time.monotonic()
    m = checked_manifest(root, resolutions)
    event('verified_runtime', source_id=m['source_id'], seconds=time.monotonic()-started)
    from src.models.neuralgcm_residual.backbone import FrozenBackbone
    from src.models.neuralgcm_residual.data import prepare_era5_pair
    from src.models.neuralgcm_residual.io import sha256, write_json
    event('imports_ready', seconds=time.monotonic()-started)
    models = {rid: FrozenBackbone.load(cp['path'], cp['sha256']).model for rid, cp in m['checkpoints'].items()}
    event('models_ready', seconds=time.monotonic()-started)
    for index in indices:
        month = m['months'][index]
        out = root / 'shards' / month['id']
        if month_complete(root, month['id'], resolutions):
            continue
        write_json(root / f'lane_{lane_id}.json', {'status':'running', 'month':month['id'],
                   'pid':os.getpid(), 'runtime':str(RUNTIME), 'log':str(root/'logs'/f'local-lane-{lane_id}.log')})
        event('month_start', month=month['id'], resolutions=resolutions)
        report = prepare_era5_pair(m['data_source'], models, {rid:out/rid for rid in models}, month['start'], month['end'])
        write_json(out/'profile.json', report)
        for rid in models:
            write_json(out/f'COMPLETE.{rid}.json', {'month':month['id'], 'source_id':m['source_id'],
                       'ready':{rid:sha256(out/rid/'READY.json')}}, immutable=True)
        event('month_complete', month=month['id'])
    write_json(root/f'lane_{lane_id}.json', {'status':'complete', 'runtime':str(RUNTIME)})


def supervise(root, workers, threads, resolutions):
    cpus = sorted(os.sched_getaffinity(0))
    if workers < 1 or threads < 1 or workers*threads > len(cpus):
        raise ValueError('Requested workers/threads exceed available affinity')
    m = checked_manifest(root, resolutions)
    from src.models.neuralgcm_residual.io import write_json
    lock = (root/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending = [i for i,x in enumerate(m['months']) if not month_complete(root,x['id'],resolutions)]
    history = root/'logs'/f'lane-history-{time.time_ns()}'
    previous = [*root.glob('lane_*.json'), *(root/'logs').glob('local-lane-*.log')]
    if previous:
        history.mkdir()
        # GPFS metadata latency on the staging host is high. These independent
        # renames retain old diagnostics without serializing every round trip.
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda path: path.rename(history/path.name), previous))
    children = set()
    stopping = threading.Event()

    def stop(signum, frame):
        stopping.set()
        for child in list(children):
            child.terminate()
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    event('supervisor_start', workers=workers, threads_per_worker=threads, pending_months=len(pending), resolutions=resolutions)
    write_json(root/'active_download.json', {'resolutions':resolutions,'workers':workers,'threads_per_worker':threads,
               'runtime':str(RUNTIME),'source_id':m['source_id'],'pid':os.getpid(),
               'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def process(i):
        indices = pending[i::workers]
        if not indices:
            return 0
        # Bound the burst of backend initialization and remote metadata requests.
        if stopping.wait(i * 0.5):
            return 130
        affinity = cpus[i*threads:(i+1)*threads]
        env = dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
                   OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
                   MKL_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads))
        log = root/'logs'/f'local-lane-{i}.log'
        cmd = ['taskset','-c',','.join(map(str,affinity)),sys.executable,'-u',str(Path(__file__).resolve()),
               'lane','--root',str(root),'--lane-id',str(i),'--indices',','.join(map(str,indices)),
               '--resolutions',*resolutions]
        for attempt in range(3):
            if stopping.is_set():
                return 130
            with log.open('a') as f:
                child = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
                children.add(child)
                code = child.wait()
                children.discard(child)
            event('lane_exit', lane=i, attempt=attempt+1, returncode=code)
            if code == 0:
                return 0
            if stopping.wait(10):
                return 130
        return code

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        codes = list(pool.map(process,range(workers)))
    write_json(root/'local_supervisor_result.json', {'codes':codes,'workers':workers,'threads':threads})
    if any(codes):
        raise RuntimeError('Some lanes failed; inspect logs and resume')
    import numpy as np
    from src.models.neuralgcm_residual.data import merge_prepared, PreparedStore
    expected=np.arange(np.datetime64('2015-01-01T00'),np.datetime64('2024-01-01T00'),np.timedelta64(6,'h'))
    for rid in resolutions:
        merge_prepared([root/'shards'/month['id']/rid for month in m['months']],root/rid)
        np.testing.assert_array_equal(PreparedStore(root/rid).times,expected)
        write_json(root/f'READY.{rid}.json',{'source_id':m['source_id'],'timestamps':len(expected),'months':108},immutable=True)
    event('download_complete', resolutions=resolutions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['verify','benchmark','lane','supervise'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=20)
    p.add_argument('--threads-per-worker', type=int, default=4)
    p.add_argument('--indices', default='')
    p.add_argument('--lane-id', type=int, default=0)
    p.add_argument('--benchmark-root', type=Path)
    p.add_argument('--frames', type=int, default=3)
    p.add_argument('--resolutions', nargs='+', choices=['res2p8','res1p4'], default=['res2p8'])
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    os.environ['JAX_PLATFORMS']='cpu'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if a.command == 'verify':
        m=checked_manifest(a.root,a.resolutions)
        event('verified_runtime', source_id=m['source_id'], python=sys.executable, libraries=m['libraries'])
        import jax
        event('jax_ready', devices=str(jax.devices()))
    elif a.command == 'lane':
        lane(a.root,[int(i) for i in a.indices.split(',') if i],a.lane_id,a.resolutions)
    elif a.command == 'benchmark':
        if a.benchmark_root is None or a.frames < 1:
            p.error('benchmark requires --benchmark-root and positive --frames')
        benchmark(a.root,a.benchmark_root,a.frames,a.resolutions)
    else:
        supervise(a.root,a.workers,a.threads_per_worker,a.resolutions)


if __name__ == '__main__':
    main()
