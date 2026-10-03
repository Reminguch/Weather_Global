"""Actual-schedule representative pilot plus exact separate-process resume."""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.campaign / 'plan.json').read_text())
    source = Path(plan['source_root'])
    sys.path.insert(0, str(source))
    output = args.campaign / 'schedule_smoke'
    output.mkdir(exist_ok=False)
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm GPU allocation required')
    child_env = dict(os.environ, JAX_PLATFORMS='cuda')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import jax
    import numpy as np
    from experimental.weather_only.train import source_hashes
    from src.models.neuralgcm_residual.io import write_json, sha256
    start = time.monotonic()
    pilot_stats = Path(plan['preflight']).with_name('pilot_statistics.json')
    common = [sys.executable, '-u', str(source / 'experimental/weather_only/train.py'),
              '--resources', plan['resources'], '--preflight', plan['preflight'],
              '--statistics', str(pilot_stats), '--pilot', '--mode', 'adaptive',
              '--updates', '6', '--validate-every', '3', '--validation-origins', '2',
              '--calibration-batches', '1', '--window', '4', '--interval', '2',
              '--peak-lr', str(plan['peak_lr']), '--warmup', str(plan['warmup'])]

    def run(directory, extra, label):
        print(json.dumps({'stage': label}), flush=True)
        with (output / (label + '.log')).open('w') as handle:
            result = subprocess.run(common + ['--output', str(directory)] + extra,
                                    env=child_env, stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=1050)
        if result.returncode:
            raise RuntimeError(label + ' failed:\n' + (output / (label + '.log')).read_text()[-6000:])

    def load(path):
        with path.open('rb') as handle:
            return pickle.load(handle)

    continuous, resumed = output / 'continuous', output / 'resumed'
    run(continuous, [], 'continuous')
    run(resumed, ['--stop-after', '3'], 'interrupted')
    stopped = load(resumed / 'last.pkl')
    assert stopped['update'] == 3 and stopped['controller']['last_probe'] == 2
    assert not (resumed / 'DONE.json').exists()
    for name in ('metrics.jsonl', 'validation.jsonl'):
        with (resumed / name).open('ab') as handle:
            handle.write(b'{"update":999}\n{"update":')
    (resumed / 'best.pkl').write_bytes(pickle.dumps({'update': 999}))
    run(resumed, ['--resume'], 'resumed')
    left, right = load(continuous / 'last.pkl'), load(resumed / 'last.pkl')
    for name in ('params', 'optimizer', 'rng', 'best'):
        assert jax.tree_util.tree_structure(left[name]) == jax.tree_util.tree_structure(right[name])
        for x, y in zip(jax.tree_util.tree_leaves(left[name]),
                        jax.tree_util.tree_leaves(right[name]), strict=True):
            np.testing.assert_array_equal(x, y, err_msg=name)
    for name in ('identity', 'update', 'controller', 'calibration', 'baseline', 'validations'):
        assert left[name] == right[name], name
    for name in ('metrics.jsonl', 'validation.jsonl'):
        def rows(directory):
            data = [json.loads(line) for line in (directory / name).read_text().splitlines()]
            for row in data:
                row.pop('seconds', None)
            return data
        assert rows(continuous) == rows(resumed), name
    config = json.loads((continuous / 'config.json').read_text())
    assert config['optimizer'] == {'peak_lr': plan['peak_lr'], 'warmup': plan['warmup']}
    assert left['update'] == 6 and left['controller']['adjustments'] == 3
    assert not np.allclose(left['controller']['weights'], 1)
    assert left['best']['selection_score'] <= 1 and left['validations'][0]['update'] == 0
    assert load(resumed / 'best.pkl')['update'] == right['best']['update']
    assert json.loads((continuous / 'DONE.json').read_text())['backbone_frozen']
    report = dict(passed=True, pilot=True, K=2, updates=6, interruption_update=3,
                  exact_separate_process_resume=True, controller_history_exact=True,
                  torn_log_and_future_best_recovery=True, initial_candidate_included=True,
                  optimizer=config['optimizer'], source_hashes=source_hashes(),
                  resources_sha256=sha256(plan['resources']), statistics_sha256=sha256(pilot_stats),
                  preflight_sha256=sha256(plan['preflight']),
                  inherited_numerical_smoke=plan['preflight'],
                  slurm_job_id=os.environ['SLURM_JOB_ID'], seconds=time.monotonic() - start)
    write_json(output / 'report.json', report, immutable=True)
    print(json.dumps({'passed': True, 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
