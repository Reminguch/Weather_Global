"""Prepare, but do not submit, the isolated 2k LR-schedule control."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'logs/ngcm_weather_only_20261002_h200_2k'
K20 = ROOT / 'logs/ngcm_weather_k20_20261003_h200_2k'
CAMPAIGN = ROOT / 'logs/ngcm_weather_lr_control_20261003_h200_2k'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    CAMPAIGN.mkdir(exist_ok=False)
    workflow = CAMPAIGN / 'workflow_v1'
    shutil.copytree(REFERENCE / 'workflow_v1', workflow)
    plan = json.loads((REFERENCE / 'plan.json').read_text())
    plan.update(modes=['adaptive'], peak_lr=1e-4, warmup=200,
                lifecycle_report=str(CAMPAIGN / 'schedule_smoke/report.json'),
                reference_campaign=str(REFERENCE),
                numerical_source_reuse='Byte-identical completed strict-mask K2 source',
                changed_training_factors={'peak_lr': [0.002, 0.0001],
                                          'warmup': [2000, 200]},
                partition_authorization='User instruction: H200 jobs use ailab.',
                restart_policy='One-hour allocations, checkpoint near 55 minutes, '
                               'same-job requeue; no 500-update cap; six slices maximum.')
    write(CAMPAIGN / 'plan.json', plan)
    source = Path(plan['source_root'])
    old_smoke = json.loads((REFERENCE / 'lifecycle_v1/report.json').read_text())
    for name, expected in old_smoke['source_hashes'].items():
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Reference numerical source changed: ' + name)
    driver = workflow / 'run_h200_slices.py'
    body = driver.read_text()
    old = "'--output',str(output)]"
    new = "'--output',str(output),'--peak-lr',str(plan['peak_lr']),\n" + \
          "          '--warmup',str(plan['warmup'])]"
    assert body.count(old) == 1
    body = body.replace(old, new)
    old = "if config['updates']!=budget or config['mode']!=mode:"
    new = "if (config['updates']!=budget or config['mode']!=mode or\n" + \
          "        config['optimizer']!={'peak_lr':1e-4,'warmup':200}):"
    assert body.count(old) == 1
    driver.write_text(body.replace(old, new))
    # Retain the K2 evaluator/reporter. Only the watcher needs a single-arm,
    # requeue-capable lifecycle, already used by the K20 campaign.
    shutil.copy2(K20 / 'workflow_v1/live_eval/watch.py', workflow / 'live_eval/watch.py')
    report = workflow / 'live_eval/report.py'
    body = report.read_text().replace('三组', '各组')
    report.write_text(body)
    shutil.copy2(Path(__file__).with_name('schedule_smoke.py'), workflow / 'schedule_smoke.py')
    shutil.copy2(Path(__file__).with_name('README.md'), CAMPAIGN / 'README.md')
    for name in ('run_h200.sbatch', 'live_eval/run_eval.sbatch'):
        body = (REFERENCE / name).read_text().replace(str(REFERENCE), str(CAMPAIGN))
        target = CAMPAIGN / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    body = (K20 / 'live_eval/run_watch.sbatch').read_text().replace(str(K20), str(CAMPAIGN))
    (CAMPAIGN / 'live_eval/run_watch.sbatch').write_text(body)
    evaluation = json.loads((REFERENCE / 'live_eval/plan.json').read_text())
    evaluation.update(campaigns={'H200': str(CAMPAIGN)}, output=str(CAMPAIGN / 'live_eval'),
                      code=str(workflow / 'live_eval'))
    write(CAMPAIGN / 'live_eval/plan.json', evaluation)
    for path in workflow.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    for path in CAMPAIGN.rglob('*.sbatch'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    result = subprocess.run(['python3', str(driver), '--campaign', str(CAMPAIGN),
                             '--mode', 'adaptive', '--dry-run'],
                            check=True, text=True, capture_output=True)
    dry = json.loads(result.stdout)
    args = dry['worker']
    assert args[args.index('--peak-lr') + 1] == '0.0001'
    assert args[args.index('--warmup') + 1] == '200'
    assert args[args.index('--updates') + 1] == '2000'
    write(CAMPAIGN / 'driver_dry_run.json', dry)
    files = [p for p in CAMPAIGN.rglob('*') if p.is_file()]
    write(CAMPAIGN / 'workflow_manifest.json', {
        str(p.relative_to(CAMPAIGN)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)})
    print(json.dumps({'campaign': str(CAMPAIGN), 'source_files_verified': len(old_smoke['source_hashes']),
                      'driver_explicit_schedule_verified': True, 'submitted': False}))


if __name__ == '__main__':
    main()
