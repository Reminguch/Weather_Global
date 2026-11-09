# Development — 2023 extreme-event evaluation

Updated: 2026-11-07 (America/New_York)

## Implementation

- `src/analysis/extremes_climatology.py`: twelve CPU workers for three observables
  by four hours. Historical reference maps are reusable after checksum validation;
  completed threshold arrays are published atomically. Finalization verifies all
  outputs and produces a hashed metadata manifest.
- `src/analysis/extremes_observer.py`: streaming observer using the maintained
  evaluator hook. Saves per-initialization occurrence/intensity sufficient
  statistics with all model, threshold and mask provenance.
- `scripts/analyze_models/report_v24_extremes_2023.py`: complete-coverage checks,
  paired block intervals, CSV/JSON/Markdown and PNG/PDF figures.
- `scripts/experiments/run_v24_extremes_2023.py` and adjacent Slurm wrapper:
  preparation, threshold jobs, immutable manifest binding, GPU evaluation and report.

## Validation

Seventeen tests passed:

```bash
source scripts/graphcast_env.sh
JAX_PLATFORMS=cpu python -m pytest -q tests/v24_Ilya/test_extremes_climatology.py tests/test_extremes_observer.py tests/v24_Ilya/test_extremes_report.py
```

Real metadata/grid/time coverage and bounded historical-field reads passed.
November7 handoff inspection found the completed production report for122 starts
and40 leads, with its checksum matching `complete.json`. Lianghong should review
the existing occurrence/intensity outputs; see [e001](experiments.md#e001) and the
[handoff README](../../../docs/experiments/README_LH.md).
No training code or live prior jobs were modified.

## Resume

Campaign on Della:
`/scratch/gpfs/DABANIN/iv9432/Weather_global/artifacts/evaluations/v24_extremes_2023_20261106`.

```bash
squeue -j 17016908,17016909,17016910,17016911
sacct -j 17016908,17016909,17016910,17016911 --format=JobID,State,Elapsed,MaxRSS -P
```

Thresholds precede binding; inference uses bound manifests under `shards/00`–`07`.
Do not edit frozen source or bound manifests in place. Check `submissions.json`
before any recovery to avoid duplicate forecasts. The submit command intentionally
refuses an existing submission record. Preserve partial outputs for diagnosis.

The report publishes to
`plots/analyze_models/data/resolution_eval/v24_extremes_2023_20261106/` and
`plots/analyze_models/images/resolution_eval/v24_extremes_2023_20261106/`,
then writes the campaign's `complete.json`. These outputs are now present and
the report checksum was verified during the November7 handoff. Scientific
interpretation remains pending; the queue commands above are for diagnostics,
not instructions to repeat the completed campaign.
