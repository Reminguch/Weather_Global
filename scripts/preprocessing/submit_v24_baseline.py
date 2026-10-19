#!/usr/bin/env python3
"""Freeze and submit the complete baseline pilot/release/production workflow."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.frozen_runtime import freeze_source, frozen_command, verify_source


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def absolute(workspace: Path, path: Path | str) -> Path:
    path = Path(path)
    return (path if path.is_absolute() else workspace / path).resolve()


def verify_workflow(pilot_root: Path, *, config: Path | None = None,
                    output_root: Path | None = None, source: Path | None = None) -> dict:
    """Check the original pilot runtime; never manufacture provenance later."""
    path = pilot_root.resolve() / 'workflow.json'
    if not path.is_file():
        raise ValueError(f'Missing frozen pilot workflow: {path}. Prepare a new pilot with '
                         'scripts/preprocessing/submit_v24_baseline.py; an old pilot cannot '
                         'be rebound to current workspace source.')
    record = json.loads(path.read_text())
    if record.get('format') != 'v24_baseline_workflow_v1':
        raise ValueError('Unsupported baseline workflow format')
    if Path(record['pilot_root']) != pilot_root.resolve():
        raise ValueError('Pilot root differs from frozen workflow')
    runtime = Path(record['source'])
    if runtime != pilot_root.resolve() / 'source':
        raise ValueError('Baseline workflow source must be the original pilot snapshot')
    snapshot = verify_source(runtime)
    if snapshot.get('workspace') != record['workspace']:
        raise ValueError('Baseline workflow workspace differs from its source snapshot')
    if sha(runtime / 'snapshot.json') != record['source_manifest_sha256']:
        raise ValueError('Baseline workflow source manifest changed')
    saved_config = Path(record['config'])
    if saved_config != pilot_root.resolve() / 'frozen_config.json':
        raise ValueError('Baseline workflow must use its pinned configuration')
    if sha(saved_config) != record['config_sha256']:
        raise ValueError('Baseline workflow configuration changed')
    for supplied, field in ((config, 'config'), (output_root, 'output_root'), (source, 'source')):
        if supplied is not None and supplied.resolve() != Path(record[field]):
            raise ValueError(f'Baseline workflow {field} differs from the pilot')
    return record


def prepare(workspace: Path, config: Path, pilot_root: Path, output_root: Path, *,
            num_shards: int = 8, initial_dependency: str | None = None) -> dict:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if initial_dependency is not None and not all(part.isdigit() for part in initial_dependency.split(":")):
        raise ValueError("initial_dependency must contain numeric job IDs separated by colons")
    workspace = workspace.resolve()
    config, pilot_root, output_root = [absolute(workspace, p) for p in (config, pilot_root, output_root)]
    if pilot_root == output_root:
        raise ValueError('Pilot and production require different cache roots')
    pilot_root.mkdir(parents=True, exist_ok=True)
    with (pilot_root / '.workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (pilot_root / 'workflow.json').exists():
            record = verify_workflow(pilot_root, output_root=output_root)
            if (record['workspace'] != str(workspace)
                    or record['original_config'] != str(config)
                    or record['original_config_sha256'] != sha(config)
                    or record.get('num_shards', 8) != num_shards
                    or record.get('initial_dependency') != initial_dependency):
                raise ValueError('Existing workflow belongs to another workspace/configuration; use new cache roots')
            return record
        if any(p.name != '.workflow.lock' for p in pilot_root.iterdir()):
            raise ValueError('Pilot root already contains artifacts without a frozen workflow; use a new root')
        if output_root.exists() and any(output_root.iterdir()):
            raise ValueError('Production root is not empty; use a new root for a new pilot')
        original = config.read_bytes()
        payload = json.loads(original)
        for field in ('prepared_root', 'anchor_manifest_root', 'baseline_checkpoint', 'stats_dir'):
            if field in payload['data']:
                payload['data'][field] = str(absolute(workspace, payload['data'][field]))
        payload['output']['output_root'] = str(absolute(workspace, payload['output']['output_root']))
        runtime = pilot_root / 'source'
        freeze_source(workspace, runtime)
        frozen_config = pilot_root / 'frozen_config.json'
        write(frozen_config, payload)
        record = {
            'format': 'v24_baseline_workflow_v1', 'workspace': str(workspace),
            'source': str(runtime), 'source_manifest_sha256': sha(runtime / 'snapshot.json'),
            'original_config': str(config), 'original_config_sha256': hashlib.sha256(original).hexdigest(),
            'config': str(frozen_config), 'config_sha256': sha(frozen_config),
            'pilot_root': str(pilot_root), 'output_root': str(output_root),
            'num_shards': num_shards, 'initial_dependency': initial_dependency,
        }
        write(pilot_root / 'workflow.json', record)
        return verify_workflow(pilot_root)


def submit(arguments: list[str]) -> str:
    command = ['sbatch', '--parsable', *arguments]
    print(shlex.join(command), flush=True)
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise ValueError(f'Unexpected sbatch response: {result.stdout!r}')
    return job


def submit_workflow(record: dict) -> dict:
    pilot_root = Path(record['pilot_root'])
    record = verify_workflow(pilot_root)
    workspace, source = Path(record['workspace']), Path(record['source'])
    (workspace / 'logs').mkdir(exist_ok=True)
    with (pilot_root / '.workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = pilot_root / 'workflow_submission.json'
        submitted = json.loads(path.read_text()) if path.exists() else {}
        if submitted and submitted.get('workflow_sha256') != sha(pilot_root / 'workflow.json'):
            raise ValueError('Submission belongs to another frozen workflow')
        submitted['workflow_sha256'] = sha(pilot_root / 'workflow.json')
        if 'pilot_job_id' not in submitted:
            submitted['pilot_job_id'] = submit([
                f'--chdir={workspace}',
                *([f"--dependency=afterok:{record['initial_dependency']}"] if record.get('initial_dependency') else []),
                str(source / 'scripts/experiments/precompute_v24_baseline.slurm'),
                'pilot', record['config'], record['pilot_root'], str(workspace), str(source), record['pilot_root'],
            ])
            write(path, submitted)
        if 'release_job_id' not in submitted:
            submitted['release_job_id'] = submit([
                f'--chdir={workspace}', f"--dependency=afterok:{submitted['pilot_job_id']}",
                str(source / 'scripts/experiments/submit_v24_baseline_after_pilot.slurm'),
                record['config'], record['output_root'], record['pilot_root'], submitted['pilot_job_id'],
                str(workspace), str(source),
            ])
            write(path, submitted)
        return submitted


def run_stage(pilot_root: Path, phase: str, config: Path, output_root: Path, shard_index: int) -> None:
    record = verify_workflow(pilot_root, config=config, source=ROOT)
    expected = record['pilot_root' if phase == 'pilot' else 'output_root']
    if output_root.resolve() != Path(expected):
        raise ValueError('Stage output differs from frozen baseline workflow')
    args = ['--config', record['config'], '--output-root', expected]
    if phase == 'pilot':
        args += ['--resume', '--num-shards', '1', '--shard-index', '0', '--max-chunks', '4', '--pilot-parity']
    elif phase == 'generate':
        args += ['--resume', '--num-shards', str(record.get('num_shards', 8)), '--shard-index', str(shard_index)]
    elif phase == 'verify':
        args += ['--num-shards', str(record.get('num_shards', 8)), '--verify']
    else:
        raise ValueError(f'Unsupported baseline phase: {phase}')
    command, kwargs = frozen_command(Path(record['source']), Path(record['workspace']),
                                     'scripts/preprocessing/precompute_v24_baseline.py', args)
    subprocess.run(command, check=True, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    prepare_parser = subparsers.add_parser('prepare')
    prepare_parser.add_argument('--workspace', type=Path, default=Path(os.environ.get('WEATHER_WORKSPACE', ROOT)))
    prepare_parser.add_argument('--config', type=Path, required=True)
    prepare_parser.add_argument('--pilot-root', type=Path, required=True)
    prepare_parser.add_argument('--output-root', type=Path, required=True)
    prepare_parser.add_argument('--submit', action='store_true')
    prepare_parser.add_argument('--num-shards', type=int, default=8)
    prepare_parser.add_argument('--initial-dependency', help='Colon-separated prerequisite job IDs')
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--pilot-root', type=Path, required=True)
    run_parser.add_argument('--phase', choices=('pilot', 'generate', 'verify'), required=True)
    run_parser.add_argument('--config', type=Path, required=True)
    run_parser.add_argument('--output-root', type=Path, required=True)
    run_parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'prepare':
        record = prepare(args.workspace, args.config, args.pilot_root, args.output_root,
                         num_shards=args.num_shards, initial_dependency=args.initial_dependency)
        print(json.dumps(record, indent=2), flush=True)
        if args.submit:
            print(json.dumps(submit_workflow(record), indent=2), flush=True)
    else:
        run_stage(args.pilot_root, args.phase, args.config, args.output_root, args.shard_index)


if __name__ == '__main__':
    main()
