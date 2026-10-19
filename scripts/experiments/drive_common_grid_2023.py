#!/usr/bin/env python3
"""Continue the authorized common-grid campaign after full staging and pilots.

The controller freezes its own two control files separately from the original
experiment snapshot. It never replaces completed evidence, cancels jobs, or
retries an ambiguous submission. Forecasts and reports execute frozen code.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

CONTROL_FILES = ('drive_common_grid_2023.py', 'common_grid_2023_merge.slurm')
TERMINAL_FAILURE = {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
                    'BOOT_FAIL', 'DEADLINE', 'PREEMPTED', 'REVOKED', 'SPECIAL_EXIT'}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def object_hash(value, *, compact=False):
    options = {'sort_keys': True, 'allow_nan': False}
    if compact:
        options['separators'] = (',', ':')
    return hashlib.sha256(json.dumps(value, **options).encode()).hexdigest()


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError('Explicit UTC timestamps are required')
    return parsed


def utc_text(value):
    return value.isoformat().replace('+00:00', 'Z')


def now():
    return utc_text(datetime.now(timezone.utc))


def write_state(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def immutable_file(path, content):
    """Create a read-only artifact once, or require the identical existing bytes."""
    path = Path(path)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(f'Immutable control artifact differs: {path}')
        return
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def install_control(workflow):
    control = workflow / 'control'
    control.mkdir(exist_ok=True)
    files = {}
    for name in CONTROL_FILES:
        source = Path(__file__).with_name(name)
        content = source.read_bytes()
        immutable_file(control / name, content)
        files[name] = hashlib.sha256(content).hexdigest()
    manifest = {'schema_version': 1, 'protocol_sha256': digest(workflow / 'protocol.json'),
                'files': files, 'scope': 'Continue this previously authorized evaluation only'}
    immutable_file(control / 'manifest.json', (json.dumps(manifest, indent=2) + '\n').encode())
    return control


def verify_protocol(workflow):
    protocol = read(workflow / 'protocol.json')
    if object_hash({k: v for k, v in protocol.items() if k != 'protocol_hash'}) != protocol.get('protocol_hash'):
        raise ValueError('Frozen protocol hash differs')
    if (protocol.get('experiment') != 'graphcast_common_grid_2023'
            or protocol.get('schema_version') != 1 or protocol.get('target_steps') != 40
            or protocol.get('lead_hours') != list(range(6, 241, 6))):
        raise ValueError('Unsupported evaluation protocol')
    starts = protocol['initialization_times']
    expected = [utc_text(datetime(2023, 1, 1, tzinfo=timezone.utc) + timedelta(days=3 * index))
                for index in range(122)]
    if starts != expected:
        raise ValueError('Expected the frozen 122 initialization dates')
    if digest(protocol['initialization_manifest']) != protocol['initialization_manifest_sha256']:
        raise ValueError('Initialization manifest changed')
    if read(protocol['initialization_manifest']).get('initializations') != starts:
        raise ValueError('Initialization manifest coverage changed')
    source = Path(protocol['source_root'])
    spec = importlib.util.spec_from_file_location('common_grid_frozen_runtime', source / 'scripts/experiments/frozen_runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.verify_source(source)['source_digest'] != protocol['source_digest']:
        raise ValueError('Frozen source identity differs from protocol')
    return protocol


def verify_staging(protocol, progress=print):
    """Validate every expected file against the full-campaign readiness index."""
    root, source = Path(protocol['data_root']), Path(protocol['source_root'])
    report = read(root / 'staging_report.json')
    manifest = read(root / 'staging_manifest.json')
    ready = read(root / '.READY.json')
    config = manifest['configuration']
    config_hash = object_hash(config, compact=True)
    if any(value.get('configuration_sha256') != config_hash for value in (manifest, report, ready)):
        raise ValueError('Staging configuration identity differs')
    if (report.get('schema_version') != 1 or report.get('status') != 'complete'
            or report.get('mode') != 'both' or report.get('limit_initializations') is not None
            or report.get('limit_truth_times') is not None
            or report.get('requested_count') != 1614
            or report.get('verified_existing_count', 0) + report.get('newly_completed_count', 0) != 1614):
        raise ValueError('Full data staging has not completed successfully')
    if (manifest.get('expected_inputs025_count') != 122 or manifest.get('expected_common_truth_count') != 1492
            or ready.get('schema_version') != 1 or ready.get('input_count') != 122
            or ready.get('truth_count') != 1492 or ready.get('manifest_sha256') != digest(root / 'staging_manifest.json')):
        raise ValueError('Staging completeness index differs')
    if (config.get('initialization_times') != protocol['initialization_times']
            or config.get('lead_hours') != protocol['lead_hours']
            or set(config.get('diagnostics', {})) != set(protocol['verification']['diagnostics'])
            or config.get('stager_sha256') != digest(source / 'scripts/preprocessing/stage_common_grid_2023.py')
            or config.get('regrid_code_sha256') != digest(source / 'scripts/analyze_models/common_grid_metrics.py')):
        raise ValueError('Staged data do not match the frozen source or evaluation coverage')
    starts = [timestamp(value) for value in protocol['initialization_times']]
    truth_times = sorted({start + timedelta(hours=lead) for start in starts for lead in protocol['lead_hours']})
    if len(truth_times) != 1492:
        raise ValueError('Unexpected common-truth valid-time coverage')
    expected = {}
    for kind, stamps in [('inputs025', starts), ('common_truth', truth_times)]:
        for stamp in stamps:
            path = (root / kind / (stamp.strftime('%Y%m%dT%H%M%SZ') + '.nc')).resolve()
            expected[str(path)] = (kind, utc_text(stamp))
    entries = ready.get('files', [])
    if len(entries) != len(expected) or {entry.get('path') for entry in entries} != set(expected):
        raise ValueError('Readiness index contains missing, duplicate, or unexpected files')
    truth_hashes = {}
    for index, entry in enumerate(entries, 1):
        path = Path(entry['path'])
        kind, stamp = expected[str(path)]
        receipt = read(str(path) + '.READY.json')
        identity = {'schema_version': 1, 'kind': kind, 'timestamp': stamp, 'path': str(path),
                    'configuration_sha256': config_hash, 'source_uri': config['source_uri'],
                    'source_structure_sha256': config['source_structure_sha256'],
                    'static_source_time': config['static_source_time']}
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise ValueError(f'Staging receipt provenance differs: {path}')
        if any(receipt.get(key) != entry.get(key) for key in ('kind', 'timestamp', 'path', 'size_bytes', 'sha256')):
            raise ValueError(f'Staging receipt differs from complete index: {path}')
        if (path.stat().st_size != receipt['size_bytes'] or digest(path) != receipt['sha256']):
            raise ValueError(f'Staged file content differs: {path}')
        if kind == 'common_truth':
            truth_hashes[stamp] = receipt['sha256']
        elif receipt.get('input_times') != [utc_text(timestamp(stamp) - timedelta(hours=6)), stamp]:
            raise ValueError(f'Native input timestamps differ: {path}')
        if index % 16 == 0 or index == len(entries):
            progress(f'[{now()}] Verified staged files {index}/{len(entries)}', flush=True)
    return {'configuration_sha256': config_hash, 'ready_sha256': digest(root / '.READY.json'),
            'manifest_sha256': digest(root / 'staging_manifest.json'),
            'report_sha256': digest(root / 'staging_report.json'),
            'input_count': 122, 'truth_count': 1492, 'truth_hashes': truth_hashes}


def staging_process_command(pid):
    try:
        raw = Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:
        return None
    return [part.decode() for part in raw.split(b'\0') if part] or None


def wait_staging(protocol, pid, on_wait=None):
    """Watch only the specified frozen staging process; never signal it."""
    if type(pid) is not int or pid <= 0:
        raise ValueError('A positive staging process id is required')
    expected_script = str(Path(protocol['source_root']) / 'scripts/preprocessing/stage_common_grid_2023.py')
    expected_output = str(Path(protocol['data_root']).resolve())
    report_path = Path(protocol['data_root']) / 'staging_report.json'
    while True:
        report = read(report_path) if report_path.exists() else {}
        command = staging_process_command(pid)
        if command is not None:
            output = None
            for index, argument in enumerate(command):
                if argument == '--output-root' and index + 1 < len(command):
                    output = command[index + 1]
                elif argument.startswith('--output-root='):
                    output = argument.split('=', 1)[1]
            if expected_script not in command or output is None or str(Path(output).resolve()) != expected_output:
                raise ValueError(f'PID {pid} is not this campaign\'s frozen staging command')
        if report.get('status') == 'failed':
            raise RuntimeError(f'Staging failed: {report.get("error", "no error detail recorded")}')
        if report.get('status') == 'complete':
            # The report is written immediately before the final readiness
            # index. Wait for the index if the producer is still finishing.
            if (Path(protocol['data_root']) / '.READY.json').exists() or command is None:
                return
        if command is None:
            raise RuntimeError(f'Staging process {pid} exited before a complete staging report')
        if on_wait is not None:
            on_wait(report)
        done = report.get('verified_existing_count', 0) + report.get('newly_completed_count', 0)
        print(f'[{now()}] Waiting for staging PID {pid}: status={report.get("status", "starting")}, '
              f'files={done}/{report.get("requested_count", 1614)}', flush=True)
        time.sleep(30)


def rss_gib(value):
    if not value:
        return 0.0
    match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)([KMGTPE]?)', value.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f'Unrecognized Slurm RSS: {value!r}')
    number, unit = float(match[1]), match[2].upper()
    # Slurm's size suffixes are binary; an unsuffixed MaxRSS is bytes.
    return number * (1024 ** (' KMGTPE'.index(unit) if unit else 0)) / 1024**3


def parse_accounting(output, job):
    rows = []
    for line in output.splitlines():
        fields = line.split('|')
        if len(fields) == 7 and fields[-1] == '':
            fields.pop()
        if not line.strip():
            continue
        if len(fields) != 6:
            raise ValueError(f'Unexpected sacct row: {line}')
        job_id, raw_id, state, exit_code, elapsed, memory = fields
        rows.append({'job_id': job_id, 'raw_id': raw_id,
                     'state': state.split()[0].rstrip('+'), 'exit_code': exit_code,
                     'elapsed_seconds': int(elapsed), 'max_rss_gib': rss_gib(memory)})
    allocation = [row for row in rows if '.' not in row['job_id']]
    if not allocation:
        return None
    if any(not re.fullmatch(re.escape(str(job)) + r'(?:_0)?', row['job_id']) for row in allocation):
        raise ValueError('Pilot accounting includes an unexpected array element')
    failed = [row for row in allocation if row['state'] in TERMINAL_FAILURE]
    if failed:
        raise RuntimeError(f'Pilot {job} failed: {failed}')
    if any(row['state'] != 'COMPLETED' for row in allocation):
        return {'complete': False, 'rows': rows}
    if any(row['exit_code'] != '0:0' for row in rows):
        raise RuntimeError(f'Pilot {job} has a nonzero accounting exit code')
    maximum = max(row['max_rss_gib'] for row in rows)
    if maximum <= 0:
        return {'complete': False, 'rows': rows, 'reason': 'awaiting completed MaxRSS accounting'}
    return {'complete': True, 'rows': rows, 'max_rss_gib': maximum,
            'elapsed_seconds': max(row['elapsed_seconds'] for row in rows),
            'accepted_job_ids': sorted({row['job_id'] for row in allocation}
                                       | {row['raw_id'] for row in allocation})}


def accounting(job):
    result = subprocess.run(['sacct', '-j', str(job), '--array', '--noheader', '--parsable2',
                             '--format=JobID,JobIDRaw,State%32,ExitCode,ElapsedRaw,MaxRSS'],
                            text=True, capture_output=True, check=True, timeout=60)
    return parse_accounting(result.stdout, job)


def wait_pilots(jobs):
    while True:
        results = {model: accounting(job) for model, job in jobs.items()}
        if all(value and value.get('complete') for value in results.values()):
            return results
        print(f'[{now()}] Waiting for pilot completion: '
              + ', '.join(f'{model}={"complete" if value and value.get("complete") else "pending/running/accounting"}'
                          for model, value in results.items()), flush=True)
        time.sleep(30)


def verify_pilot(workflow, protocol, model, job, account, truth_hashes):
    path = workflow / 'pilot' / f'{model}_00.json'
    result = read(path)
    smoke = workflow / 'initializations_smoke.json'
    start = protocol['initialization_times'][0]
    if read(smoke).get('initializations') != [start]:
        raise ValueError('Pilot manifest must contain only the first frozen initialization')
    required = {'schema_version': 1, 'status': 'complete', 'model': model,
                'protocol_hash': protocol['protocol_hash'], 'protocol_sha256': digest(workflow / 'protocol.json'),
                'original_protocol_hash': protocol['original_protocol_hash'],
                'initialization_manifest_sha256': digest(smoke), 'shard_index': 0, 'shard_count': 1}
    if any(result.get(key) != value for key, value in required.items()):
        raise ValueError(f'{model} pilot provenance differs')
    if str(result.get('slurm_job_id')) not in set(account['accepted_job_ids']) | {str(job)}:
        raise ValueError(f'{model} pilot does not belong to the specified Slurm job')
    records = result.get('per_initialization', [])
    if len(records) != 1 or records[0].get('initialization_time') != start:
        raise ValueError(f'{model} pilot must contain exactly the initial date')
    record = records[0]
    valid = [utc_text(timestamp(start) + timedelta(hours=hour)) for hour in protocol['lead_hours']]
    if (record.get('lead_hours') != protocol['lead_hours'] or record.get('valid_times') != valid
            or record.get('truth_sha256') != [truth_hashes[value] for value in valid]):
        raise ValueError(f'{model} pilot forecast leads or verification truth differ')
    branches = {'gc1', 'm1'} if model == 'm1' else {'gc025'}
    if set(record.get('mse', {})) != branches:
        raise ValueError(f'{model} pilot has unexpected score branches')
    for diagnostics in record['mse'].values():
        if set(diagnostics) != set(protocol['verification']['diagnostics']):
            raise ValueError(f'{model} pilot diagnostics differ')
        for values in diagnostics.values():
            if len(values) != 40 or any(not isinstance(v, (float, int)) or isinstance(v, bool)
                                        or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError(f'{model} pilot has incomplete or invalid scores')
    evidence = result.get('evidence', {})
    expected_checkpoint = protocol['m1']['checkpoint_sha256'] if model == 'm1' else protocol['gc025_checkpoint_sha256']
    if evidence.get('checkpoint_sha256') != expected_checkpoint:
        raise ValueError(f'{model} pilot checkpoint differs')
    if model == 'm1' and evidence.get('baseline_checkpoint_sha256') != protocol['gc1']['checkpoint_sha256']:
        raise ValueError('M1 pilot backbone checkpoint differs')
    for key in ('elapsed_seconds', 'peak_rss_gib'):
        if not isinstance(result.get(key), (int, float)) or not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f'{model} pilot lacks a valid {key}')
    return {'path': str(path), 'sha256': digest(path), 'job_id': str(job),
            'elapsed_seconds': max(result['elapsed_seconds'], account['elapsed_seconds']),
            'peak_rss_gib': max(result['peak_rss_gib'], account['max_rss_gib']),
            'accounting': account}


def resource_budget(model, pilot):
    peak, elapsed = pilot['peak_rss_gib'], pilot['elapsed_seconds']
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
               for value in (peak, elapsed)):
        raise ValueError('Finite positive pilot resources are required')
    if model == 'm1':
        if peak * 1.25 > 16:
            raise ValueError('M1 pilot leaves insufficient memory headroom for the authorized 16G production shape')
        return {'shards': 8, 'mem_gib': 16, 'walltime_seconds': 3720, 'walltime': '01:02:00'}
    if model != 'gc025':
        raise ValueError('Unknown production model')
    memory = max(16, 8 * math.ceil(1.5 * peak / 8))
    seconds = 900 * math.ceil(max(3720, 1.4 * elapsed * math.ceil(122 / 16)) / 900)
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    walltime = (f'{days}-' if days else '') + f'{hours:02d}:{minutes:02d}:00'
    return {'shards': 16, 'mem_gib': memory, 'walltime_seconds': seconds, 'walltime': walltime}


def find_production_records(workflow, model):
    result = []
    for path in sorted((workflow / 'submissions').glob('*.json')):
        record = read(path)
        if record.get('phase') == 'production' and record.get('model') == model:
            result.append(record)
    return result


def bounded_submit(command, *, cwd, timeout=120):
    """Bound scheduler network stalls, including the helper's sbatch child.

    A timed-out submission remains ambiguous and is never automatically retried.
    Only the local command process group is terminated; no Slurm job cancellation
    command is issued. Any successfully recorded job is retained for recovery.
    """
    process = subprocess.Popen(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=10)
        stderr += ('\nController submission timeout; acceptance may be ambiguous. '
                   'Reconcile the recorded submission intent and Slurm jobs before any retry.\n')
        return subprocess.CompletedProcess(command, 124, stdout, stderr)


def production_submission(workflow, protocol, model, budget, state, state_path):
    operations = state.setdefault('operations', {})
    existing = find_production_records(workflow, model)
    if len(existing) > 1:
        raise RuntimeError(f'Multiple existing production submissions for {model}; no new jobs submitted')
    if existing:
        record = existing[0]
        if (record.get('shards') != budget['shards'] or record.get('mem') != f"{budget['mem_gib']}G"
                or record.get('time') != budget['walltime']):
            raise RuntimeError(f'Existing {model} production resources differ; no duplicate submitted')
        operations[model] = {'status': 'submitted', 'job_id': record['job_id'], 'record': record}
        write_state(state_path, state)
        return record['job_id']
    if model in operations:
        raise RuntimeError(f'Ambiguous earlier {model} submission intent; reconcile its jobs before restarting')
    command = [sys.executable, '-u', str(Path(protocol['source_root']) / 'scripts/experiments/run_common_grid_2023.py'),
               'submit', '--workflow', str(workflow), '--phase', 'production', '--model', model,
               '--mem', f"{budget['mem_gib']}G", '--time', budget['walltime'], '--shards', str(budget['shards'])]
    operations[model] = {'status': 'submitting', 'started_at': now(), 'command': command}
    write_state(state_path, state)
    print(f'[{now()}] Submitting {model}: {command}', flush=True)
    result = bounded_submit(command, cwd=protocol['workspace'])
    operations[model]['stdout'], operations[model]['stderr'] = result.stdout, result.stderr
    operations[model]['returncode'] = result.returncode
    write_state(state_path, state)
    print(result.stdout, end='', flush=True)
    if result.returncode:
        raise RuntimeError(f'Production submission failed for {model}: {result.stderr}')
    # Frozen submit writes its job receipt atomically. Recovery can discover
    # that receipt if this process stops before saving the returned job id.
    records = find_production_records(workflow, model)
    if len(records) != 1:
        raise RuntimeError(f'Cannot identify the submitted {model} job unambiguously')
    operations[model].update(status='submitted', job_id=records[0]['job_id'], record=records[0])
    write_state(state_path, state)
    return records[0]['job_id']


def merge_submission(workflow, protocol, jobs, state, state_path):
    operations = state.setdefault('operations', {})
    if 'merge' in operations:
        if operations['merge'].get('status') == 'submitted':
            return operations['merge']['job_id']
        raise RuntimeError('Ambiguous earlier merge submission intent; reconcile its job before restarting')
    command = ['sbatch', '--parsable', '--job-name=cgrid23-merge',
               '--dependency=afterok:' + ':'.join(str(jobs[model]) for model in ('m1', 'gc025')),
               '--kill-on-invalid-dep=yes', f'--output={workflow}/logs/merge_%j.out',
               f'--error={workflow}/logs/merge_%j.err',
               str(workflow / 'control/common_grid_2023_merge.slurm'), str(workflow)]
    operations['merge'] = {'status': 'submitting', 'started_at': now(), 'command': command}
    write_state(state_path, state)
    result = bounded_submit(command, cwd=protocol['workspace'])
    operations['merge'].update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
    write_state(state_path, state)
    if result.returncode:
        raise RuntimeError(f'Merge submission failed: {result.stderr}')
    job = result.stdout.strip().split(';')[0]
    if not re.fullmatch(r'\d+', job):
        raise RuntimeError(f'Unrecognized merge job identifier: {result.stdout}')
    operations['merge'].update(status='submitted', job_id=job)
    write_state(state_path, state)
    return job


def drive(workflow, jobs, staging_pid=None):
    control = workflow / 'control'
    with (control / 'controller.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another controller already owns this campaign') from None
        state_path = control / 'state.json'
        protocol = verify_protocol(workflow)
        state = read(state_path) if state_path.exists() else {
            'schema_version': 1, 'protocol_hash': protocol['protocol_hash'], 'pilot_jobs': jobs,
            'created_at': now(), 'operations': {}, 'errors': [],
        }
        if state.get('protocol_hash') != protocol['protocol_hash'] or state.get('pilot_jobs') != jobs:
            raise ValueError('Controller state belongs to a different protocol or pilot job pair')
        if state.get('status') == 'production_and_merge_submitted':
            print(json.dumps(state['production_jobs'], indent=2), flush=True)
            return
        try:
            if staging_pid is not None:
                def record_staging_wait(report):
                    state.update(status='waiting_for_staging', staging_pid=staging_pid,
                                 staging_progress=report, updated_at=now())
                    write_state(state_path, state)
                record_staging_wait({})
                wait_staging(protocol, staging_pid, on_wait=record_staging_wait)
            state.update(status='verifying_full_staging', updated_at=now())
            write_state(state_path, state)
            staged = verify_staging(protocol)
            truth_hashes = staged.pop('truth_hashes')
            state['staging'] = staged
            state['status'] = 'waiting_for_pilots'
            write_state(state_path, state)
            accounts = wait_pilots(jobs)
            pilots = {model: verify_pilot(workflow, protocol, model, jobs[model], accounts[model], truth_hashes)
                      for model in ('m1', 'gc025')}
            budgets = {model: resource_budget(model, pilots[model]) for model in pilots}
            state.update(pilots=pilots, production_budgets=budgets, status='submitting_production')
            write_state(state_path, state)
            production = {model: production_submission(workflow, protocol, model, budgets[model], state, state_path)
                          for model in ('m1', 'gc025')}
            production['merge'] = merge_submission(workflow, protocol, production, state, state_path)
            state.update(status='production_and_merge_submitted', production_jobs=production, updated_at=now())
            write_state(state_path, state)
            print(json.dumps({'status': state['status'], 'jobs': production, 'budgets': budgets}, indent=2), flush=True)
        except BaseException as error:
            state.setdefault('errors', []).append({'at': now(), 'type': type(error).__name__, 'message': str(error)})
            state.update(status='stopped_with_error', updated_at=now())
            write_state(state_path, state)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', required=True, type=Path)
    parser.add_argument('--m1-pilot-job', required=True)
    parser.add_argument('--gc025-pilot-job', required=True)
    parser.add_argument('--staging-pid', type=int,
                        help='Wait for this frozen staging process on della-vis1 before full verification')
    args = parser.parse_args(argv)
    if socket.gethostname().split('.')[0] != 'della-vis1':
        parser.error('Run this controller on della-vis1 after the full staging command succeeds')
    if not all(re.fullmatch(r'\d+', job) for job in (args.m1_pilot_job, args.gc025_pilot_job)):
        parser.error('Pilot identifiers must be numeric Slurm array job ids')
    if args.staging_pid is not None and args.staging_pid <= 0:
        parser.error('--staging-pid must be positive')
    workflow = args.workflow.resolve()
    control = install_control(workflow)
    if Path(__file__).resolve() != (control / CONTROL_FILES[0]).resolve():
        command = [sys.executable, '-u', str(control / CONTROL_FILES[0]),
                   '--workflow', str(workflow), '--m1-pilot-job', args.m1_pilot_job,
                   '--gc025-pilot-job', args.gc025_pilot_job]
        if args.staging_pid is not None:
            command += ['--staging-pid', str(args.staging_pid)]
        os.execv(sys.executable, command)
    drive(workflow, {'m1': args.m1_pilot_job, 'gc025': args.gc025_pilot_job}, args.staging_pid)


if __name__ == '__main__':
    main()
