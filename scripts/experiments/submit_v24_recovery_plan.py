"""Submit an explicitly prepared recovery graph, preserving each submission intent."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    digest = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    args.ledger.mkdir(parents=True, exist_ok=True)
    with (args.ledger / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.ledger / 'jobs.json'
        state = json.loads(path.read_text()) if path.exists() else {
            'plan': str(args.plan.resolve()), 'plan_sha256': digest, 'jobs': {}}
        if state['plan_sha256'] != digest:
            raise ValueError('Submission plan changed; use a new ledger')
        seen = set()
        for node in plan['nodes']:
            if node['id'] in seen or any(d not in seen for d in node['depends_on']):
                raise ValueError('Graph must have unique IDs in dependency order')
            seen.add(node['id'])
            if node['command'][0] != 'sbatch' or '--parsable' not in node['command']:
                raise ValueError('Only explicit parsable sbatch commands are supported')
            if any(x.startswith('--array=') and '%' in x for x in node['command']):
                raise ValueError('Recovery arrays must be unthrottled')
        for node in plan['nodes']:
            command = list(node['command'])
            if node['depends_on']:
                dependencies = [state['jobs'].get(d, {}).get('job_id', f'<{d}>') for d in node['depends_on']]
                command[1:1] = [f"--dependency={node.get('dependency_condition', 'afterok')}:" + ':'.join(dependencies),
                                '--kill-on-invalid-dep=yes']
            print(shlex.join(command), flush=True)
            if not args.submit:
                continue
            old = state['jobs'].get(node['id'])
            if old:
                if old['command'] != command or not old.get('job_id'):
                    raise ValueError(f"Changed or unresolved submission: {node['id']}; inspect Slurm")
                print(node['id'], old['job_id'], '(already submitted)', flush=True)
                continue
            state['jobs'][node['id']] = {'command': command, 'workflow': node['workflow'],
                'source_digest': node['source_digest'], 'status': 'submission_intent'}
            write(path, state)
            environment = dict(os.environ)
            environment.pop('JAX_PLATFORMS', None)
            result = subprocess.run(command, check=True, text=True, capture_output=True, env=environment, cwd=ROOT)
            job = result.stdout.strip().split(';')[0]
            if not job.isdigit():
                raise ValueError(f'Unexpected sbatch output: {result.stdout}')
            state['jobs'][node['id']].update(job_id=job, status='submitted', stderr=result.stderr)
            write(path, state)
            print(node['id'], job, flush=True)


if __name__ == '__main__':
    main()
