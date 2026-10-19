#!/usr/bin/env python3
"""Identify the archived green curves from public score checksums and raster fit."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import xarray as xr
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / '.cache/presentation-python'))
import pymupdf


def checksum(path, algorithm='sha256'):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, algorithm)
    return base64.b64encode(digest.digest()).decode() if algorithm == 'md5' else digest.hexdigest()


def main():
    workflow = ROOT / 'artifacts/evaluations/graphcast_common_grid_2023_20261015'
    audits = workflow / 'audits'
    protocol = json.loads((workflow / 'protocol.json').read_text())
    archived = Path('/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/wb2_scores')
    gc_path = archived / 'graphcast_vs_era5_240x121_2022.nc'
    ens_path = archived / 'ens_mean_vs_era5_240x121_2022.nc'
    official = json.loads((audits / 'wb2_graphcast_listing.json').read_text())['items']
    gc_meta = next(r for r in official if r['name'] == 'benchmark_results/' + gc_path.name)
    ens_meta = json.loads((audits / 'wb2_ens_metadata.json').read_text())
    checkpoint = Path(protocol['gc025_checkpoint'])
    dm = json.loads((audits / 'deepmind_checkpoint_listing.json').read_text())['items']
    ck_meta = next(r for r in dm if Path(r['name']).name == checkpoint.name)
    identities = []
    sources = [(gc_path, gc_meta), (ens_path, ens_meta), (checkpoint, ck_meta)]
    stats_meta = json.loads((audits / 'deepmind_stats_listing.json').read_text())['items']
    for name in ('diffs_stddev_by_level.nc', 'mean_by_level.nc', 'stddev_by_level.nc'):
        sources.append((Path(protocol['stats_dir']) / name,
                        next(r for r in stats_meta if r['name'] == 'stats/' + name)))
    for path, meta in sources:
        md5 = checksum(path, 'md5')
        if md5 != meta['md5Hash'] or path.stat().st_size != int(meta['size']):
            raise ValueError('Official source mismatch: ' + str(path))
        identities.append({'path': str(path), 'sha256': checksum(path), 'md5_base64': md5,
                           'official_bucket': meta['bucket'], 'official_object': meta['name'],
                           'official_generation': meta['generation'], 'official_md5_match': True})
    # Retain small, checksum-verified score inputs in our own audit folder.
    recovered = audits / 'recovered_wb2'
    recovered.mkdir(exist_ok=True)
    for path in (gc_path, ens_path):
        dest = recovered / path.name
        if dest.exists():
            if checksum(dest) != checksum(path):
                raise ValueError('Archived local copy changed')
        else:
            dest.write_bytes(path.read_bytes())
    g = xr.load_dataset(gc_path, decode_timedelta=True).sel(region='global')
    e = xr.load_dataset(ens_path, decode_timedelta=True).sel(region='global')
    summary = json.loads((workflow / 'report/summary.json').read_text())
    days = g.lead_time.values / np.timedelta64(1, 'D')
    np.testing.assert_array_equal(days, np.asarray(summary['lead_hours']) / 24)
    tree = ast.parse((ROOT / 'Overleaf/tools/rebuild_common_grid_figures.py').read_text())
    geometry = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ('LABELS', 'YTICKS')}
    figure = ROOT / 'Overleaf/tools/figure_label_sources/Graphs/add_mamba.pdf'
    with pymupdf.open(figure) as document:
        images = document[0].get_images()
        if len(images) != 1:
            raise ValueError('Expected one archived raster')
        pixels = np.asarray(Image.open(io.BytesIO(document.extract_image(images[0][0])['image'])).convert('RGB'))
    if pixels.shape != (1540, 2520, 3):
        raise ValueError('Changed archived geometry')
    from common_grid_metrics import DIAGNOSTICS
    records = {}
    for index, key in enumerate(geometry['LABELS']):
        spec = DIAGNOSTICS[key]
        gc_score = g['rmse.' + spec.variable]
        ens_score = e['rmse.' + spec.variable].sel(lead_time=g.lead_time)
        if spec.level is not None:
            gc_score = gc_score.sel(level=spec.level)
            ens_score = ens_score.sel(level=spec.level)
        ratio = 100 * gc_score.values / ens_score.values
        row, col = divmod(index, 4)
        left, top = 98 + 618 * col, 97 + 491 * row
        crop = pixels[top:top + 386, left:left + 536].astype(float)
        mask = np.linalg.norm(crop - [31, 158, 78], axis=-1) <= 12
        if index == 0:
            mask[105-top:179-top, 105-left:431-left] = False
        slope, intercept = np.polyfit([p for v, p in geometry['YTICKS'][index]],
                                     [v for v, p in geometry['YTICKS'][index]], 1)
        xs, ys = [], []
        for x in range(3, 533):
            hits = np.flatnonzero(mask[:, x])
            if len(hits) and np.ptp(hits) <= 12:
                xs.append(x)
                ys.append(float(np.median(hits)) + top)
        if len(xs) < 100:
            raise ValueError('Insufficient curve coverage: ' + key)
        trace_days = (np.asarray(xs) - .5) / 534 * 10
        trace = np.interp(days, trace_days, slope * np.asarray(ys) + intercept)
        visible = (days >= 1) & (days <= 9.75)
        j = summary['diagnostics'].index(key)
        current = np.asarray(summary['rmse']['gc025'])[:, j]
        records[key] = {'published_2022_gc025_rmse': gc_score.values.tolist(),
                        'published_2022_ens_rmse': ens_score.values.tolist(),
                        'published_2022_gc025_percent_ens': ratio.tolist(),
                        'raster_green_curve_percent_ens_approx': trace.tolist(),
                        'raster_vs_published_rms_difference_percentage_points_days1_to9p75':
                            float(np.sqrt(np.mean((trace[visible] - ratio[visible])**2))),
                        'production_2023_gc025_rmse': current.tolist(),
                        'production2023_vs_published2022_rmse_change_pct': (100 * (current / gc_score.values - 1)).tolist()}
    result = {'status': 'archived_reference_score_source_identified',
              'script_sha256': checksum(__file__), 'archived_figure_sha256': checksum(figure),
              'official_source_identities': identities, 'lead_days': days.tolist(), 'diagnostics': records,
              'conclusions': [
                  'The archived green curves match the published WeatherBench 2022 GraphCast-versus-ERA5 scores on the 1.5-degree grid, divided by published 2022 ENS-mean-versus-ERA5 scores.',
                  'The new 2023 reference checkpoint is byte-identical to DeepMind public GraphCast ERA5 1979-2017, resolution 0.25, 37-level weights.',
                  'Published 2022 scores and the locally rerun 2023 forecasts are distinct reference sources. Their checkpoint equivalence has not been established.',
                  'The separately found local GraphCast-operational evaluation is not the recovered source of these twelve archived reference curves.'
              ], 'limits': [
                  'Raster identification is approximate; file identity against the public bucket is exact by byte count and MD5.',
                  'The public 2022 score file contains no checkpoint identifier or initialization manifest. The website FAQ describes a through-2019 GraphCast version, but this does not uniquely establish the 2022 score checkpoint.',
                  'Different-year score differences do not establish an inference defect or isolate a causal effect of checkpoint training coverage.',
                  'Independent upstream forecast audit is a separate test; no production fields, metrics, or manuscript figures are modified.'
              ]}
    output = audits / 'reference_provenance.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for key, record in records.items():
        print(key, 'raster/source RMS pp', round(record['raster_vs_published_rms_difference_percentage_points_days1_to9p75'], 3))
    print(output)


if __name__ == '__main__':
    main()
