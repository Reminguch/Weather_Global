# Full-GC comparison eligibility and the 3e-7 learning rate

Updated: 2026-11-12 (America/New_York).

The user explicitly rejected the full-GC `1e-5` training setting and requested
that its results never be brought back into scientific comparisons. Preserve
the checkpoints, logs, evaluations and frozen source as historical evidence;
exclude these runs from default tables, plots, rankings, method contrasts and
scientific conclusions. This is an explicit user decision about this setting,
not an execution failure or an automatic gate on finite loss deterioration.

## Excluded lineage

Campaign: `artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/`.

| Actual run ID | Lambda | Status |
| --- | ---: | --- |
| `gcft_s22_l000_lr0` | 0 | Rejected `1e-5` calibration; diagnostic archive only |
| `gcft_s22_l025` | 0.25 | Excluded from comparisons |
| `gcft_s22_l050_profile` | 0.5 | Excluded from comparisons |
| `gcft_s22_l075` | 0.75 | Excluded from comparisons |
| `gcft_s22_l100` | 1 | Excluded from comparisons |

The exclusion follows all checkpoints, continuations, SWAs and filtered
forecasts derived from these runs. Raw numerical evidence and warnings remain
unchanged. Only an explicit request to diagnose this rejected setting should
retrieve its scores. The separate `3e-5` calibration is also historical tuning
evidence, not the selected full-GC comparison. GC–Mamba runs are not affected.

The current reporting implementation is
[`comparison_policy.py`](../../scripts/analyze_models/comparison_policy.py).
Frozen submitted runtimes are preserved; an old runtime can still write an
archival all-arm report. Such a report is not an eligible comparison. Apply the
current policy when presenting it, and use the current filtered presentation
instead. Do not alter a frozen source or sealed scientific report in place.

Current presentation: [eligible pilot report](../../plots/analyze_models/data/resolution_eval/v24_raw_checkpoint_pilot_20261110_eligible_20261112_v2/report.md)
and [eligible MSE/power plot](../../plots/analyze_models/images/resolution_eval/v24_raw_checkpoint_pilot_20261110_eligible_20261112_v2/mse_power_tradeoff.png).
The original campaign also records the exact excluded configuration hashes in
[`comparison_status_20261112.json`](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/comparison_status_20261112.json).

## Accepted full-GC rate and available evidence

Use **constant LR `3e-7` from the first update, without warmup or decay**, for
future full-GC fine-tuning. The [constant-rate protocol](v24_full_gc_constant_lr3em7_2026-11-12.md)
supersedes the earlier cosine replacement and starts all five arms fresh.
The existing historical run is `gcft_s22_l000_lr3em7_fresh_20261107`, lambda 0;
it used peak3e-7 with cosine decay to3e-8 and100-update warmup. It remains
eligible as clearly labeled historical evidence, not a constant-rate result.
Its native checkpoints exist at updates 0/250/500/750/1000/1250.

| Checkpoint | Evaluation cohort | Exact rollout GC loss | Public GC loss | Reduction |
| --- | --- | ---: | ---: | ---: |
| 500 | 32 development starts in 2022 | 10.337334 | 10.840716 | 4.6434% |
| 1250 | Four development starts in 2022, 40 leads | 10.610624 | 11.133143 | 4.6934% |

These are different cohorts, not a learning curve. At step1250, day-10 loss
reduction is 8.0881%; aggregate spherical power mismatch is 63.972% versus
public GC 61.571%. Finite reference-repeat differences remain warnings. The
step1250 decrease exceeds the measured baseline spread of 0.0793851, which
does not estimate full training or forecast uncertainty.

Sources: [step500 evaluation](../../artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107/calibration_evaluation/gcft_s22_l000_lr3em7_fresh_20261107/evaluation.json)
and the eligible lambda-zero row in the [archived step1250 measurements](../../plots/analyze_models/data/resolution_eval/v24_raw_checkpoint_pilot_20261110/summary.csv).
No positive-lambda full-GC `3e-7` result or update2500 checkpoint exists at this
inspection. The saved-checkpoint diagnostic17404694/17404732 subsequently
completed; its training schedule remains historical cosine.

## Submitted settings inspected on November 12

| Work | Peak LR | State at inspection |
| --- | --- | --- |
| Full-GC lambda0, `17049163_0` | `3e-7` | Pending continuation, 1250 to 2500 |
| Full-GC positive lambda, `17049163_1–4` | `1e-5` | Pending; results excluded |
| New M1-matched GC–Mamba, `17406861/17406862` | Spatial `1e-4`; main Mamba multiplier 3 | Pending; separate valid recipe |
| Portable fresh full-GC K8/K12/K16/K20 | `3e-7`, all lambda0 | Prepared; remote submissions unknown |

The existing full-GC positive-lambda submissions therefore do **not** use
`3e-7`. Their saved1250 weights cannot be reused as fresh low-rate runs.
The mixed final chain `17406897/17406898/17406899` still references these old
lineages, so its eventual presentation must apply the exclusion too.

The user subsequently **authorized replacement** of the four pending high-rate
continuations with fresh low-rate runs. Follow the
[replacement protocol](v24_full_gc_positive_lambda_lr3em7_restart_2026-11-12.md)
for the historical first replacement. A later user instruction supersedes it:
all five full-GC arms now start fresh at constant3e-7, including lambda0.
Arrays17429091→17429092 replace all pending cosine training; the original
lambda0 checkpoints are archived unchanged. GC–Mamba's training schedule is
intact. The table above records the pre-replacement inspection, not the current
configuration; follow the constant-rate protocol for active jobs.
