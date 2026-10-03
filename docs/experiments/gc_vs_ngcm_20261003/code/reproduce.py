"""Recompute reported RMSE improvements from archived evaluator JSON, on CPU."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NGCM = ROOT.parent / 'weather_only_20261003'
GC_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def read(path):
    return json.loads(path.read_text())


def rms(values):
    return math.sqrt(sum(x*x for x in values) / len(values))


def improvement(now, baseline):
    assert math.isfinite(now) and math.isfinite(baseline) and baseline > 0
    return 100 * (1 - now / baseline)


def main():
    records, source_hashes, gc_data = [], {}, {}

    def load(path):
        source_hashes[str(path.relative_to(ROOT.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read(path)

    for horizon in (2, 22):
        data = load(ROOT / 'evidence/gc' / f'v22_K{horizon}_K40.json')
        gc_data[horizon] = data
        assert data['n_samples'] == 32 and data['target_steps'] == 40
        for field, values in data['per_variable_per_step'].items():
            for lead in (12, 120, 240):
                index = lead // 6 - 1
                ref, now = values['rmse_baseline'][index], values['rmse_full'][index]
                value = improvement(now, ref)
                assert abs(value - values['improvement_pct_rmse'][index]) < 1e-9
                records.append(dict(model='GraphCast', checkpoint=f'v22_K{horizon}_step26000',
                                    field=field, band='native GC task levels', lead_hours=lead,
                                    baseline_rmse=ref, rmse=now, improvement_percent=value))

    baseline_differences = {}
    for field, values in gc_data[2]['per_variable_per_step'].items():
        other = gc_data[22]['per_variable_per_step'][field]['rmse_baseline']
        baseline_differences[field] = max(abs(x-y) / x for x, y in zip(values['rmse_baseline'], other))

    reference = None
    for mode, step in (('adaptive', 2000), ('adaptive', 1840), ('calibrated', 2000)):
        data = load(NGCM / 'results' / f'H200_{mode}_step{step}.json')
        baseline = load(NGCM / 'baselines' / Path(data['baseline_path']).name)
        assert data['finite'] and len(data['origins']) == 16
        assert data['origins'] == baseline['signature']['origins']
        identity = (data['origins'], data['pressure_hpa'], baseline['physical_rmse'])
        if reference is None:
            reference = identity
        else:
            assert identity == reference, 'NGCM reference arrays or origins differ'
        for band, levels in [('weather_gt30hpa', [p for p in data['pressure_hpa'] if p > 30]),
                             ('common_gc_13levels', GC_LEVELS)]:
            indices = [data['pressure_hpa'].index(p) for p in levels]
            for field, values in data['physical_rmse'].items():
                for lead in (12, 120):
                    index = data['lead_hours'].index(lead)
                    now = rms([values[index][j] for j in indices])
                    ref = rms([baseline['physical_rmse'][field][index][j] for j in indices])
                    records.append(dict(model='NeuralGCM', checkpoint=f'K2_{mode}_step{step}',
                                        field=field, band=band, lead_hours=lead,
                                        baseline_rmse=ref, rmse=now,
                                        improvement_percent=improvement(now, ref)))
    counts = []
    keys = sorted({(r['model'], r['checkpoint'], r['band'], r['lead_hours']) for r in records})
    for model, checkpoint, band, lead in keys:
        values = [r['improvement_percent'] for r in records
                  if (r['model'], r['checkpoint'], r['band'], r['lead_hours']) ==
                     (model, checkpoint, band, lead)]
        counts.append(dict(model=model, checkpoint=checkpoint, band=band, lead_hours=lead,
                           improved=sum(v > .1 for v in values), worse=sum(v < -.1 for v in values),
                           unchanged=sum(abs(v) <= .1 for v in values)))
    result = dict(metric='100 * (1 - full physical RMSE / own frozen baseline RMSE)',
                  display_tolerance_percent=0.1, records=records, counts=counts,
                  gc_K2_K22_max_relative_baseline_difference=baseline_differences,
                  gc_raw_json_anchor_ids_available=False,
                  causal_head_to_head_between_models=False, source_sha256=source_hashes)
    (ROOT / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(records)
    (ROOT / 'comparison.csv').write_text(stream.getvalue())
    print(json.dumps({'records': len(records), 'sources': len(source_hashes),
                      'ngcm_baseline_arrays_exactly_match': True}))


if __name__ == '__main__':
    main()
