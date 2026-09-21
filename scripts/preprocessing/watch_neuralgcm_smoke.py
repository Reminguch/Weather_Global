#!/usr/bin/env python3
"""Durable, fail-closed 2.8-degree download -> one GPU smoke test watcher.

Only stdlib is imported while waiting. No GPU is allocated until all 13,148
frames have been verified. Submission intent is persisted before sbatch:
after an ambiguous crash an operator must reconcile it, never blindly resubmit.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def expected_times():
    start, end = datetime(2015, 1, 1), datetime(2024, 1, 1)
    return [(start + timedelta(hours=6*i)).isoformat(timespec='hours')
            for i in range(int((end-start).total_seconds() // 21600))]


def ready(root):
    return all((root / p).is_file() for p in
               ('READY.res2p8.json', 'res2p8/READY.json', 'res2p8/manifest.json'))


def metadata(root, source_id):
    if not ready(root):
        raise ValueError('The complete 2.8-degree readiness markers are absent')
    marker = read(root / 'READY.res2p8.json')
    if marker != dict(source_id=source_id, timestamps=13148, months=108):
        raise ValueError('Completion marker identity/count mismatch')
    plan = read(root / 'download_manifest.json')
    if plan['source_id'] != source_id:
        raise ValueError('Download source identity changed')
    if [m['id'] for m in plan['months']] != [f'{y}-{m:02}' for y in range(2015, 2024) for m in range(1, 13)]:
        raise ValueError('Download month plan changed')
    for month in plan['months']:
        if not (root / 'shards' / month['id'] / 'res2p8/READY.json').is_file():
            raise ValueError('Missing completed month: ' + month['id'])
    path = root / 'res2p8/manifest.json'
    identity = sha(path)
    if read(path.parent / 'READY.json')['manifest_sha256'] != identity:
        raise ValueError('Merged manifest SHA256 mismatch')
    manifest = read(path)
    if sorted(r['time'] for r in manifest['records']) != expected_times():
        raise ValueError('Missing, duplicate or unexpected six-hour timestamps')
    grid = manifest['data_grid']
    if tuple(len(grid[k]) for k in ('longitude', 'latitude', 'levels')) != (128, 64, 37):
        raise ValueError('Not the expected 2.8-degree grid')
    return manifest, identity


def verify_frame(root, record):
    path = (root / 'res2p8' / record['file']).resolve()
    if root.resolve() not in path.parents or 'res2p8' not in path.parts:
        raise ValueError('Frame is outside the selected scratch dataset: ' + str(path))
    if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
        raise ValueError('Frame size/hash mismatch: ' + str(path))


def verify_dataset(root, source_id):
    manifest, identity = metadata(root, source_id)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(lambda r: verify_frame(root, r), manifest['records']):
            pass
    if sha(root / 'res2p8/manifest.json') != identity:
        raise ValueError('Manifest changed during verification')
    dataset_id = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                                          allow_nan=False).encode()).hexdigest()
    return dict(passed=True, manifest_sha256=identity, source_id=source_id, dataset_id=dataset_id,
                frames=13148, checked_at=time.time())


def check_snapshot(run):
    for rel, identity in read(run / 'snapshot.json').items():
        if sha(run / rel) != identity:
            raise ValueError('Smoke snapshot changed: ' + rel)


def stage(project, root, run):
    plan = read(root / 'download_manifest.json')
    run.mkdir(parents=True, exist_ok=False)
    (run / 'logs').mkdir()
    code = run / 'code'
    shutil.copytree(project / 'src', code / 'src',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for vendor in ('graphcast', 'neuralgcm'):
        shutil.copytree(project / 'third_party' / vendor, code / 'third_party' / vendor,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', 'build', '*.egg-info'))
    for rel in ('scripts/preprocessing/watch_neuralgcm_smoke.py',
                'scripts/training/smoke_neuralgcm_training.py',
                'scripts/training/smoke_neuralgcm_after_download.sbatch'):
        dst = code / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / rel, dst)
    cp = plan['checkpoints']['res2p8']
    dest = run / 'deterministic_2_8_deg.pkl'
    shutil.copy2(cp['path'], dest)
    if sha(dest) != cp['sha256']:
        raise ValueError('Checkpoint differs from pinned download identity')
    config = dict(data_root=str(root.resolve()), source_id=plan['source_id'],
                  gpu_python=str(project / '.venv_neuralgcm/bin/python'),
                  libraries=plan['libraries'])
    write(run / 'config.json', config)
    files = [p for p in code.rglob('*') if p.is_file()] + [dest, run / 'config.json']
    write(run / 'snapshot.json', {str(p.relative_to(run)): sha(p) for p in files})
    write(run / 'status.json', dict(phase='waiting_download'))
    print(json.dumps(config), flush=True)


def command(args):
    return subprocess.run(args, check=True, universal_newlines=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60).stdout.strip()


def submit(run, cfg, state, save):
    if state.get('job_id'):
        return
    if state.get('submission_intent'):
        raise RuntimeError('Ambiguous previous submission. Reconcile Slurm before restarting.')
    save(phase='submitting', submission_intent=time.time())
    job = command(['sbatch', '--parsable', '--chdir=' + str(run / 'code'),
                   '--output=' + str(run / 'logs/smoke-%j.out'),
                   '--error=' + str(run / 'logs/smoke-%j.err'),
                   str(run / 'code/scripts/training/smoke_neuralgcm_after_download.sbatch'),
                   str(run), cfg['data_root'], cfg['gpu_python']]).split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Unexpected sbatch response: ' + job)
    save(phase='submitted', job_id=job)


def track(run, state, save):
    job = state['job_id']
    queue = command(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%T|%R'])
    active = next((line for line in queue.splitlines() if line.startswith(job+'|')), '')
    if active:
        save(phase='smoke_active', scheduler=active)
        return False
    rows = command(['sacct', '-X', '-n', '-P', '-j', job, '-o', 'JobIDRaw,State,ExitCode'])
    row = next((line.split('|') for line in rows.splitlines() if line.startswith(job+'|')), None)
    if not row:
        save(phase='awaiting_accounting')
        return False
    status = row[1].split()[0].rstrip('+')
    if status in ('PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING', 'REQUEUED', 'SUSPENDED'):
        save(phase='smoke_active', scheduler=status)
        return False
    if status != 'COMPLETED' or row[2] != '0:0':
        raise RuntimeError('GPU smoke job failed: ' + '|'.join(row))
    report = read(run / 'result/report.json')
    if report.get('passed') is not True or not report.get('frozen_parameters_unchanged'):
        raise RuntimeError('Smoke report did not pass')
    if not report.get('devices') or any(not d.startswith('cuda:') for d in report['devices']):
        raise RuntimeError('Smoke did not run on CUDA')
    if report.get('kind') != 'real_era5_one_update_smoke':
        raise RuntimeError('Unexpected smoke report kind')
    if report.get('dataset_id') != read(run / 'verification.json')['dataset_id']:
        raise RuntimeError('Smoke used a different prepared dataset')
    if report.get('checkpoint_sha256') != sha(run / 'deterministic_2_8_deg.pkl'):
        raise RuntimeError('Smoke used a different checkpoint')
    save(phase='passed', report=str(run / 'result/report.json'))
    return True


def watch(run, poll):
    lock = (run / 'watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = read(run / 'status.json')
    if state['phase'] in ('passed', 'failed'):
        print('Already terminal:', state, flush=True)
        return

    def save(**fields):
        state.update(fields, heartbeat=time.time(), host=socket.gethostname(), pid=os.getpid())
        write(run / 'status.json', state)
        print(json.dumps(state, sort_keys=True), flush=True)

    try:
        check_snapshot(run)
        cfg = read(run / 'config.json')
        root = Path(cfg['data_root'])
        if state.get('submission_intent') and not state.get('job_id'):
            raise RuntimeError('Submission outcome unknown; reconcile before resubmitting')
        last_progress = time.monotonic()
        while True:
            if state.get('job_id'):
                try:
                    if track(run, state, save):
                        return
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    save(phase='scheduler_retry', scheduler_error=str(exc))
            elif ready(root):
                save(phase='verifying_all_frames')
                write(run / 'verification.json', verify_dataset(root, cfg['source_id']))
                check_snapshot(run)
                submit(run, cfg, state, save)
            else:
                count = sum(1 for _ in root.glob('shards/*/res2p8/*/frames/*.npz.json'))
                if count != state.get('downloaded_frames'):
                    last_progress = time.monotonic()
                save(phase='waiting_download', downloaded_frames=count, expected_frames=13148,
                     stalled_warning=time.monotonic()-last_progress > 1800)
            time.sleep(poll)
    except Exception as exc:
        save(phase='failed', error=repr(exc))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('stage', 'watch', 'check'))
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--root', type=Path)
    p.add_argument('--project', type=Path)
    p.add_argument('--poll-seconds', type=int, default=60)
    args = p.parse_args()
    if args.command == 'stage':
        if args.root is None or args.project is None:
            p.error('stage needs --root and --project')
        stage(args.project.resolve(), args.root.resolve(), args.run_root)
    elif args.command == 'watch':
        if args.poll_seconds < 10:
            p.error('poll-seconds must be at least 10')
        watch(args.run_root, args.poll_seconds)
    else:
        check_snapshot(args.run_root)
        cfg = read(args.run_root / 'config.json')
        _, identity = metadata(Path(cfg['data_root']), cfg['source_id'])
        checked = read(args.run_root / 'verification.json')
        if not checked['passed'] or checked['manifest_sha256'] != identity:
            raise ValueError('Full frame verification is missing or stale')
        expression = ('import importlib.metadata as m, json, sys; '
                      'pins=json.loads(sys.argv[1]); '
                      'actual={k:m.version(k) for k in pins}; '
                      'assert actual == pins, (actual,pins)')
        command([cfg['gpu_python'], '-c', expression, json.dumps(cfg['libraries'])])
        print('Frozen source, checkpoint, dataset and GPU environment validated', flush=True)


if __name__ == '__main__':
    main()
