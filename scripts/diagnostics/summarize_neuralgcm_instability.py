#!/usr/bin/env python3
"""Export compact, hash-linked evidence from completed diagnostic GPU jobs."""
import argparse
import hashlib
import json
from pathlib import Path


def compact_step(step, *, final):
    result = {}
    for key, value in step.items():
        if key.endswith('_l0'):
            continue
        if key.endswith('_area_mean_by_sigma'):
            if all(isinstance(x, (int, float)) for x in value):
                result[key + '_range'] = [min(value), max(value)]
            continue
        if key == 'fields' and not final:
            continue
        if key == 'native':
            result['surface_pressure_pa_unweighted'] = value['surface_pressure_pa']
        elif key in ('before_correction', 'after_correction'):
            result[key] = {k: v for k, v in value.items() if k != 'fields' or final}
        else:
            result[key] = value
    return result


def compact_rollouts(rollouts):
    return [dict(origin=r['origin'], mode=r['mode'], steps=[
        compact_step(s, final=i == len(r['steps']) - 1)
        for i, s in enumerate(r['steps'])]) for r in rollouts]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', type=Path, default=Path('logs/neuralgcm_instability_20260924'))
    parser.add_argument('--output', type=Path, default=Path(
        'docs/experiments/neuralgcm_residual/instability_audit_20260924.json'))
    args = parser.parse_args()
    result = {'scope': 'Epoch-5 frozen checkpoints; interventions do not retrain weights',
              'created_date_utc': '2026-09-24', 'reports': []}
    for path in sorted(args.logs.glob('*/report.json')):
        raw = path.read_bytes()
        report = json.loads(raw)
        if not report.get('completed') or not report.get('production_checkpoint_unchanged'):
            raise ValueError(f'Incomplete diagnostic or checkpoint identity check: {path}')
        item = {k: report[k] for k in ('run_id', 'job_id', 'checkpoint_sha256',
                                      'script_sha256', 'identities', 'devices', 'elapsed_seconds')}
        item.update(source_report=str(path), source_report_sha256=hashlib.sha256(raw).hexdigest())
        if 'k1_audit' in report:
            item['kind'] = 'metric_and_feedback_audit'
            item['k1_audit'] = {k: v for k, v in report['k1_audit'].items()
                                if k != 'native_samples'}
            item['impulses'] = [{k: v for k, v in s.items() if not k.startswith('native')}
                                for s in report['impulses']]
            item['variants'] = compact_rollouts(report['variants'])
        elif 'first_injections' in report:
            item['kind'] = 'degree_zero_interventions'
            # Modal arrays are available in the hash-linked original report.
            # Keep only area means and physically interpretable pressure ranges.
            item['rollouts'] = compact_rollouts(report['rollouts'])
        elif 'pressure_sweeps' in report:
            item['kind'] = 'pressure_interventions'
            item['rollouts'] = compact_rollouts(report['rollouts'])
            item.update({k: report[k] for k in (
                'pressure_sweeps', 'pressure_objective_derivatives')})
        else:
            raise ValueError(f'Unknown report type: {path}')
        result['reports'].append(item)
    if len(result['reports']) != 8:
        raise ValueError('Expected two metric audits, four pressure and two degree-zero jobs')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Exported {len(result["reports"])} completed reports to {args.output}')


if __name__ == '__main__':
    main()
