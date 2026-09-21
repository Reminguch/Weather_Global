#!/usr/bin/env python3
"""Stream public ERA5 once per timestamp into both pinned NeuralGCM grids."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True, help='Exclusive end time')
    p.add_argument('--output-root', required=True, type=Path)
    p.add_argument('--checkpoint-2p8', required=True, type=Path)
    p.add_argument('--checkpoint-1p4', required=True, type=Path)
    p.add_argument('--source', default='gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3')
    args = p.parse_args()
    from src.models.neuralgcm_residual.backbone import FrozenBackbone
    from src.models.neuralgcm_residual.data import prepare_era5_pair
    from src.models.neuralgcm_residual.io import write_json
    models = {'res2p8': FrozenBackbone.load(args.checkpoint_2p8).model,
              'res1p4': FrozenBackbone.load(args.checkpoint_1p4).model}
    report = prepare_era5_pair(args.source, models, {k: args.output_root / k for k in models}, args.start, args.end)
    write_json(args.output_root / 'profile.json', report)
    print(report)


if __name__ == '__main__':
    main()
