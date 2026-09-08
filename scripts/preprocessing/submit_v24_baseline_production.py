#!/usr/bin/env python3
"""Submit production only after the measured GPU pilot passes all checks."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.mamba.v24_Ilya.baseline_cache import (
    atomic_json, build_manifest, ensure_manifest, load_context, verify_cache,
)


def memory_gib(value: str) -> float:
    match = re.fullmatch(r'([0-9.]+)([KMGT]?)', value.strip())
    if not match:
        return 0.0
    return float(match[1]) * {'': 1 / 2**30, 'K': 1 / 2**20, 'M': 1 / 2**10, 'G': 1, 'T': 1024}[match[2]]


def production_budget(report, largest_shard, accounting_peak=0.0):
    peak = max(float(report['peak_host_gib']), accounting_peak)
    memory = max(32, math.ceil(peak * 1.5 / 8) * 8)
    seconds = 1.5 * (float(report['initialization_seconds']) + float(report['compilation_seconds_estimate']) + largest_shard * float(report['steady_chunk_seconds']))
    minutes = max(62, math.ceil(seconds / 1800) * 30)
    return memory, minutes


def submit(args):
    command = ['sbatch', '--parsable', *args]
    print(shlex.join(command), flush=True)
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise ValueError(f'Unexpected sbatch response: {result.stdout!r}')
    print(f'Submitted job {job}', flush=True)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--pilot-job-id', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.pilot_job_id.isdigit():
        raise ValueError('Expected numeric pilot job ID')
    report = json.loads((args.pilot_root / 'pilot_report.json').read_text())
    if report.get('passed') is not True or report['chunks'] != 4 or max(report['max_normalized_errors'].values()) > 1e-5:
        raise ValueError('Pilot parity did not pass for all four chunks')
    context = load_context(args.config)
    manifest = build_manifest(context, 8)
    if manifest['compatibility_sha256'] != report['compatibility_sha256']:
        raise ValueError('Pilot is incompatible with current source/configuration/code')
    pilot_manifest = json.loads((args.pilot_root / 'manifest.json').read_text())
    if pilot_manifest != build_manifest(context, 1, 4) or report['manifest_sha256'] != pilot_manifest['manifest_sha256']:
        raise ValueError('Pilot manifest does not match the required four representative chunks')
    # Verify before any production submission, including file checksums.
    verify_cache(args.pilot_root, allow_partial=True)
    accounting = subprocess.run(
        ['sacct', '-j', args.pilot_job_id, '--noheader', '--parsable2', '--format=JobID,State,MaxRSS'],
        check=True, text=True, capture_output=True).stdout
    rows = [row.split('|') for row in accounting.splitlines() if row.strip()]
    if not any(row[0] == args.pilot_job_id and row[1] == 'COMPLETED' for row in rows):
        raise ValueError(f'Pilot is not COMPLETED in accounting: {accounting}')
    peak = max([memory_gib(row[2]) for row in rows if len(row) > 2] + [0.0])
    largest = max(sum(c['shard'] == i for c in manifest['chunks']) for i in range(8))
    memory, minutes = production_budget(report, largest, peak)
    estimate = {'host_memory_gib': memory, 'time_minutes': minutes, 'num_shards': 8,
                'largest_shard_chunks': largest, 'accounting_peak_host_gib': peak,
                'pilot_job_id': args.pilot_job_id, 'pilot_report': report,
                'manifest_sha256': manifest['manifest_sha256'], 'accounting': accounting}
    print(json.dumps(estimate, indent=2), flush=True)
    if args.dry_run:
        return
    ensure_manifest(args.output_root, manifest)
    if shutil.disk_usage(args.output_root).free < manifest['prediction_bytes'] * 1.1:
        raise RuntimeError('Insufficient free space for predictions plus 10% headroom')
    # Serialize submission and save each ID before the next external action.
    import fcntl
    with (args.output_root / '.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record_path = args.output_root / 'submission.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record['manifest_sha256'] != manifest['manifest_sha256']:
                raise ValueError('Existing submission belongs to another cache')
        else:
            record = estimate
        if 'array_job_id' not in record:
            record['array_job_id'] = submit([
                '--array=0-7', f'--mem={memory}G', f'--time={minutes}',
                'scripts/experiments/precompute_v24_baseline.slurm',
                'generate', str(args.config), str(args.output_root),
            ])
            atomic_json(record_path, record)
        if 'verification_job_id' not in record:
            record['verification_job_id'] = submit([
                f"--dependency=afterok:{record['array_job_id']}",
                'scripts/experiments/verify_v24_baseline.slurm', str(args.config), str(args.output_root),
            ])
            atomic_json(record_path, record)
        print(json.dumps(record, indent=2), flush=True)
        with Path('current_experiments.md').open('a') as f:
            f.write(f"\n- Baseline cache production submission: pilot `{args.pilot_job_id}` passed; array `{record['array_job_id']}` (8 shards), verification `{record['verification_job_id']}`; {memory}G host RAM / {minutes} minutes per shard. Cache: `{args.output_root}`.\n")


if __name__ == '__main__':
    main()
