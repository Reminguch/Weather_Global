# NeuralGCM import into minimalistic_code

Imported from `origin/residual_NGCM` commit
`95b1d2d5a703e29586b716e936dc32eb99adbe86` (October 7, 2026).
The existing GraphCast source, environment, and working changes are preserved.

## Runnable experiment paths

- `experimental/weather_only/`: strict p > 30 hPa K2 training, fixed calibration
  or adaptive weights, numerical preflight and separate-process resume smoke.
- `experimental/weather_k20/`: fresh native K20 training with the same objective,
  preflight and resume smoke. This is not K2-to-K20 transfer.
- `experimental/adaptive_loss/`: shared objective, controller and checkpoint code.
- `src/models/neuralgcm_residual/`: frozen backbone, native corrections, data,
  normalization, Graph–Mamba branch, trajectory training and older pipeline tools.
- `third_party/neuralgcm/`: the branch's vendored upstream package and license.
- `experimental/weather_space_residual/README.md`: proposal only.

The starting model is deterministic 2.8-degree NeuralGCM, width128/d_inner16,
mesh4, seed22, fresh zero head, batch2. K2 supervises 6/12h; K20 supervises
6–120h with memory BPTT and stopped physical-feedback gradients. Defaults retain
the original 0.002 peak LR and 2,000-step warmup. No schedule experiment has been
silently substituted.

## Import adaptations and provenance

`configs/experiments/neuralgcm_residual/import_manifest.json` records original
source paths and SHA256 values. The October 7 numerical archive takes precedence
over older copies. Archived reports and code remain under `docs/experiments/`.
The branch's older README is `README_SOURCE_BRANCH.md`; the pre-existing README
in this workspace was preserved.

The committed branch omitted `paper_lifecycle.py`, `physical_selection.py`, and
`pressure_weighted_loss.py`, although archived scripts import them. For the
strict K2/K20 entry points, the latter two imports are unused and were removed.
A local `paper_lifecycle.lock_run` prevents concurrent writers and requires a
checkpoint for resume; checkpoint contents are validated by the original trainer.
The original adaptive campaign's train/preflight/lifecycle scripts and its
pressure-loss reference test remain archival, not runnable entry points here.
Controller/checkpoint/calibration tests were ported separately. The historical
lower-LR preparation script also remains archival because it requires the old
campaign directories.

The independent evaluator now uses `WeatherOnlyLoss` for its diagnostic scalar
instead of the unavailable pressure-loss helper. Its physical RMSE calculation
still covers all seven fields and all levels. This is a local adaptation, not a
byte-identical replay of the historical evaluator. The imported live watcher and
reporter retain the documented rendering failure; do not use them as an unattended
production workflow until that repair is tested.

## Local prerequisites

The original resource root is
`/scratch/gpfs/MENGDIW/sh4809/weatherforecast/Weather_Global`. This account cannot
read it. No prepared data, frozen-model weights, or trained residual checkpoints
were included in Git. Supply a readable resource JSON with **absolute** paths:

```json
{
  "checkpoint": "/path/to/deterministic_2_8_deg.pkl",
  "checkpoint_sha256": "actual checkpoint SHA256",
  "prepared": "/path/to/prepared/res2p8",
  "statistics": "/path/to/native_normalization.json"
}
```

The prepared store requires its manifest, READY marker and frame files. Native
normalization also requires its matching climatology NPZ. The trainer's separate
`--statistics` argument is the 24-hour **weather loss** statistics, not native
increment normalization. Full training requires the complete 2015–2023 prepared
store; training uses 2015–2021, selection uses 2022, and 2023 remains reserved.

The source environment pins JAX 0.9.2; the existing GraphCast environment has
JAX 0.9.0.1. Install dependencies in an isolated environment before GPU execution:

```bash
source scripts/graphcast_env.sh
python -m venv .venv_neuralgcm
source scripts/neuralgcm_env.sh
python -m pip install -r configs/experiments/neuralgcm_residual/requirements-cuda12.lock.txt
python scripts/experiments/check_neuralgcm_local.py --resources /absolute/resources.json
```

The checker does not load checkpoints or run forecasts. It reports prerequisite
gaps; dataset identity, all-frame checks and numerical gates still run in the
actual preflight. Archived GPU pass reports cannot replace fresh local gates.

## First GPU checks, once prerequisites are available

Submit each phase separately from the repository root:

```bash
mkdir -p logs
sbatch scripts/experiments/neuralgcm_local_smoke.sbatch k20-preflight \
  --resources /absolute/resources.json \
  --output "$PWD/artifacts/diagnostics/ngcm_local/k20_preflight"

# After preflight succeeds:
sbatch scripts/experiments/neuralgcm_local_smoke.sbatch k20-resume \
  --resources /absolute/resources.json \
  --preflight "$PWD/artifacts/diagnostics/ngcm_local/k20_preflight/report.json" \
  --statistics "$PWD/artifacts/diagnostics/ngcm_local/k20_preflight/pilot_statistics.json" \
  --output "$PWD/artifacts/diagnostics/ngcm_local/k20_resume"
```

The launcher uses H200, ailab/gpu-test, 20 GiB CPU RAM, eight CPUs and 30 minutes.
Accounting for original jobs 14937571/14937572 shows 7m19s/18m40s and peak CPU
RSS 12.30/9.33 GiB. These resources cover one smoke phase, not full training.
Use k2-preflight/k2-resume for the shorter pipeline with distinct output paths.
Pilot statistics remain separate from the full 60-timestamp training statistics.

Training CLI: `python experimental/weather_k20/train.py --help` or
`python experimental/weather_only/train.py --help`. Both enforce local preflight
and resume gates, and refuse execution outside a Slurm GPU allocation.
No jobs were submitted as part of this import.

## Saved-results audit

```bash
source scripts/graphcast_env.sh
python docs/experiments/ngcm_test_review_20261007/reproduce.py
```

This CPU-only audit verified 67 evidence files, 32 evaluator checkpoints and
3,360 field/lead/band comparisons. It does not require training data or weights.

## Import validation

135 CPU tests passed across `tests/neuralgcm_residual/` and
`experimental/weather_only/test_weather_only.py`, including controller, mask,
gradient, recovery and local locking tests. These ran in the existing GraphCast
environment with `gin-config==0.5.0` installed only into the workspace's
`.cache/neuralgcm-import-deps` directory. The GraphCast environment was not changed.
This verifies the import on CPU, not parity with the original pinned GPU runtime.

```bash
source scripts/graphcast_env.sh
JAX_PLATFORMS=cpu \
PYTHONPATH="$PWD/.cache/neuralgcm-import-deps:$PWD:$PWD/third_party/graphcast:$PWD/third_party/neuralgcm" \
python -m pytest -q tests/neuralgcm_residual experimental/weather_only/test_weather_only.py
```

`artifacts/diagnostics/ngcm_local/import_readiness.json` records the current
dependency and data gaps. No GPU numerical or resume test has run locally yet.
