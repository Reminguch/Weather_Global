"""CPU-only audit of saved evaluator JSON. Run from any working directory."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = (
    'temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind',
    'specific_humidity', 'specific_cloud_ice_water_content',
    'specific_cloud_liquid_water_content',
)
LABELS = ('Temperature', 'Geopotential', 'Zonal wind', 'Meridional wind',
          'Specific humidity', 'Cloud ice', 'Cloud liquid water')
COMMON_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
LEADS = (6, 12, 24, 72, 120)


def read(path):
    return json.loads(path.read_text())


def main():
    manifest = read(ROOT / 'artifact_manifest.json')
    for item in manifest['entries']:
        data = (ROOT / item['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
        assert len(data) == item['bytes'], item['path']

    records, inputs, checkpoints = [], {}, []
    reference = None

    def load(path):
        relative = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(Path('..') / path.relative_to(ROOT.parent))
        inputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read(path)

    cases = []
    for campaign in ('native_k20', 'k2_low_lr'):
        directory = ROOT / 'evidence' / campaign
        for path in (directory / 'results').glob('*.json'):
            cases.append((campaign, path, directory / 'baselines'))
    old = ROOT.parent / 'weather_only_20261003'
    for mode, step in (('adaptive', 1840), ('adaptive', 2000), ('calibrated', 2000)):
        cases.append(('k2_strict_' + mode,
                      old / 'results' / f'H200_{mode}_step{step}.json', old / 'baselines'))
    original = ROOT.parent / 'adaptive_loss_20261002'
    cases.append(('k2_original_adaptive', original / 'results/H200_adaptive_step2000.json',
                  original / 'baselines'))

    for campaign, path, baseline_directory in sorted(cases, key=lambda item: (item[0], read(item[1])['update'])):
        result = load(path)
        baseline = load(baseline_directory / Path(result['baseline_path']).name)
        assert result['finite'] is True, path
        assert len(result['origins']) == 16, path
        assert result['origins'] == baseline['signature']['origins'], path
        assert result['pressure_hpa'] == baseline['signature']['pressure_hpa'], path
        assert result['lead_hours'] == baseline['lead_hours'], path
        assert set(result['physical_rmse']) == set(FIELDS), path
        identity = (result['origins'], result['pressure_hpa'], baseline['physical_rmse'])
        if reference is None:
            reference = identity
        else:
            assert identity == reference, 'Different origins/levels/frozen baseline arrays: ' + str(path)
        for field in FIELDS:
            for value in result['physical_rmse'][field]:
                assert len(value) == 37 and all(math.isfinite(x) and x >= 0 for x in value)
        checkpoint = dict(campaign=campaign, update=result['update'],
                          checkpoint_sha256=result['checkpoint_sha256'],
                          source=str(path.relative_to(ROOT.parent)))
        checkpoints.append(checkpoint)
        bands = {
            'weather_gt30hpa': [i for i, p in enumerate(result['pressure_hpa']) if p > 30],
            'all37': list(range(37)),
            'common_gc13': [result['pressure_hpa'].index(p) for p in COMMON_LEVELS],
        }
        assert len(bands['weather_gt30hpa']) == 29
        for band, indices in bands.items():
            for lead in LEADS:
                index = result['lead_hours'].index(lead)
                for field in FIELDS:
                    rmse = math.sqrt(math.fsum(result['physical_rmse'][field][index][i] ** 2 for i in indices) / len(indices))
                    ref = math.sqrt(math.fsum(baseline['physical_rmse'][field][index][i] ** 2 for i in indices) / len(indices))
                    assert ref > 0
                    records.append(dict(campaign=campaign, update=result['update'],
                                        band=band, lead_hours=lead, variable=field,
                                        baseline_rmse=ref, rmse=rmse,
                                        improvement_percent=100 * (1 - rmse / ref)))

    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(records)
    (ROOT / 'comparison.csv').write_text(stream.getvalue())
    index = {(r['campaign'], r['update'], r['band'], r['lead_hours'], r['variable']):
             r['improvement_percent'] for r in records}

    def table(columns, band='weather_gt30hpa', lead=120):
        lines = ['| Variable | ' + ' | '.join(x[2] for x in columns) + ' |',
                 '| --- | ' + ' | '.join(['---:'] * len(columns)) + ' |']
        for field, label in zip(FIELDS, LABELS):
            values = [index[(campaign, step, band, lead, field)] for campaign, step, _ in columns]
            lines.append('| ' + label + ' | ' + ' | '.join(f'{x:+.2f}%' for x in values) + ' |')
        return '\n'.join(lines)

    tables = {
        'primary_120h': table([
            ('k2_strict_calibrated', 2000, 'K2 fixed calibration, 2000'),
            ('k2_strict_adaptive', 2000, 'K2 adaptive, 2000'),
            ('k2_low_lr', 2000, 'K2 lower LR, 2000'),
            ('native_k20', 1100, 'K20 selected best, 1100'),
            ('native_k20', 1620, 'K20 latest evaluated, 1620'),
        ]),
        'k20_short_12h': table([('native_k20', 1100, 'K20 best, 1100'),
                                 ('native_k20', 1620, 'K20 snapshot, 1620')], lead=12),
        'k20_all37_120h': table([('native_k20', 1100, 'K20 best, 1100'),
                                  ('native_k20', 1620, 'K20 snapshot, 1620')], band='all37'),
    }
    lines = ['| K20 update | ' + ' | '.join(LABELS) + ' |',
             '| ---: | ' + ' | '.join(['---:'] * 7) + ' |']
    k20_steps = sorted({r['update'] for r in records if r['campaign'] == 'native_k20'})
    for step in k20_steps:
        lines.append('| ' + str(step) + ' | ' + ' | '.join(
            f'{index[("native_k20", step, "weather_gt30hpa", 120, field)]:+.2f}%'
            for field in FIELDS) + ' |')
    tables['k20_checkpoint_history'] = '\n'.join(lines)
    (ROOT / 'tables.md').write_text('\n\n'.join('## ' + k + '\n\n' + v for k, v in tables.items()) + '\n')

    # Confirm that changing K did not silently change the objective/controller source.
    prior_config = load(old / 'training/adaptive/config.json')
    k20_config = load(ROOT / 'evidence/native_k20/training/config.json')
    source_equality = {}
    for relative in ('experimental/weather_only/objective.py',
                     'experimental/adaptive_loss/controller.py',
                     'experimental/adaptive_loss/trainer.py',
                     'src/models/neuralgcm_residual/paper_loss.py',
                     'src/models/neuralgcm_residual/normalization.py'):
        source_equality[relative] = prior_config['source_hashes'][relative] == k20_config['source_hashes'][relative]
        assert source_equality[relative], relative
        assert hashlib.sha256((ROOT / 'evidence/numerical_source' / relative).read_bytes()).hexdigest() == k20_config['source_hashes'][relative]

    smoke = {}
    for path in sorted((ROOT / 'evidence/smoke').glob('*.json')):
        result = load(path)
        assert result['passed'] and result['pilot']
        smoke[path.stem] = {k: result[k] for k in ('passed', 'K', 'slurm_job_id', 'updates')}
    completion = {campaign: load(ROOT / 'evidence' / campaign / 'training/DONE.json')
                  for campaign in ('native_k20', 'k2_low_lr')}
    assert completion['native_k20']['updates'] == completion['k2_low_lr']['updates'] == 2000
    assert completion['native_k20']['best_update'] == 1100
    assert 2000 not in k20_steps, 'Update the report: final K20 independent evaluation is now available'

    audit = dict(audit_date='2026-10-07', gpu_execution_in_this_audit=False,
                 copied_artifacts_verified=len(manifest['entries']),
                 evaluator_checkpoints=len(checkpoints), comparison_rows=len(records),
                 origins_count=16, frozen_baseline_arrays_and_origins_exactly_match=True,
                 source_equality=source_equality, recorded_smoke_reports=smoke,
                 native_k20_completed_updates=2000, native_k20_independent_evaluation_steps=k20_steps,
                 native_k20_final_independent_evaluation_available=False,
                 source_sha256=inputs, checkpoints=checkpoints)
    (ROOT / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: audit[k] for k in ('copied_artifacts_verified', 'evaluator_checkpoints',
                                          'comparison_rows', 'frozen_baseline_arrays_and_origins_exactly_match')}))


if __name__ == '__main__':
    main()
