# Research

Updated: 2026-11-13 (America/New_York)

**Current full-GC schedule:** all five lambda arms now start fresh at
**constant `3e-7` from the first update, with zero warmup and no decay**.
Training `17429091 → 17429092`, history `17429093` and completion `17429094`
replace the earlier cosine submissions, including the old lambda0 continuation.
No constant-rate scientific result exists yet. The saved cosine lambda0
checkpoints are historical evidence and cannot stand in for this new schedule.
The November `1e-5` full-GC lineages remain excluded from scientific comparisons.
GC–Mamba's M1-matched training/history `17406861/17406862/17406863` is unchanged.
See [e014](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#e014)
and the [current protocol](../docs/experiments/v24_full_gc_constant_lr3em7_2026-11-12.md).
The rebuilt mixed raw/SWA chain is `17429215 → 17429216 → 17429217`;
annual/smoothing controller `17429218` follows the same training dependencies,
`17429092:17406862`.

November 13 [optimal-blurring evaluation for v4](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#e015):
the 2022 fit is complete and 92/122 annual starts are evaluated (January 1–
October 1). Filtered M1 has 4.9850% lower rollout loss than filtered GC
(paired 95% interval 4.1243–5.9559%) on this subset. The remaining 30 October–
December starts in `17430310_6/7` are queued; automatic report `17430311` waits.
This is incomplete annual evidence. The [interim readout](../artifacts/evaluations/v24_optimal_smoothing_v4_20261112/interim_20261113/REPORT.md)
preserves canonical v4 loss reductions, separate spectral scores and measured
rerun drift. Existing training and constrained-smoothing campaigns are unchanged.
The [joint MSE/power comparison](../plots/analyze_models/data/resolution_eval/v24_optimal_smoothing_v4_interim92_power_20261113/REPORT.md)
shows that filtering improves MSE but worsens power matching for both models.
Raw M1 beats filtered GC on both aggregate measures; versus raw GC it improves
MSE but worsens power matching. Full-year power report `17470053` follows
`17430311`. These are still 92-start point estimates, not annual conclusions.
See the [protocol](../docs/experiments/v24_optimal_smoothing_v4_2026-11-12.md).

This index starts with the current matched memory-training comparison. Earlier
experiments remain in the existing records; this is not a historical migration.

| ID | Direction / question | Category | Status | Current conclusion | Next step |
| --- | --- | --- | --- | --- | --- |
| r001 | [Does training with memory improve the selected configuration?](directions/r001-matched-memory-training/overview.md) | training | active | Nov13 e003: five fresh local carry SWAs give 14.1682% ± 0.4332 pp rollout and 21.0283% ± 0.7502 pp day-10 reduction (mean ± seed SD). Original matched-SWA 5.261 pp [4.698,5.862] contrast retained; local raw reset control has one seed. | Finish v4 build/publication separately. Carry-seed variation is measured; matched reset-seed variation remains unmeasured. Keep original/local and SWA/raw provenance distinct. |
| r002 | [How does the selected correction compare with higher-resolution GraphCast?](directions/r002-common-grid-forecast-comparison/overview.md) | evaluation | active | Nov13 e013: existing ENS-matched Figure 5/9 scores gain paired pointwise 95% date intervals on 91–108 starts per point; all twelve day-10 M1-vs-GC025 intervals have positive lower bounds. Point estimates unchanged. Separate 2018 annual completion remains unverified. | Finish v4 build/publication with date uncertainty distinguished from seed SD. Inspect annual array15870083, report15870088 and plot15870456 before claiming 2018 paper parity; no new status check performed. |
| r003 | [Fresh residual width comparison](directions/r003-fresh-residual-width/overview.md) | training | closed | Unexecuted proposal superseded: the user clarified that the question is the benefit of larger data. | Follow r004. |
| r004 | [Does more cached training data improve each residual width?](directions/r004-cached-training-data-scale/overview.md) | data | active | Oct28:20y replacement submitted; verification16392462 passed, prepare16392486 queued; new-arm startup16392467/68 waits on cache. Both7y controls reused; obsolete pending40y chain cancelled. No data-size result. | Inspect new2002–2022 preparation/cache progression, then two20k-open/2k-closed lineages and common2023 evaluation. |
| r005 | [Can MGDA improve spectral fidelity while preserving exact GC accuracy?](directions/r005-mgda-accuracy-spectral-adjustment/overview.md) | training | active | Nov12 e014: five fresh full-GC constant3e-7/no-warmup arms submitted as17429091→17429092; history17429093/completion17429094. Earlier cosine jobs canceled before starting; no constant-rate result yet. GC–Mamba17406861→17406862 unchanged;1e-5 exclusion retained. | Inspect constant-rate full-GC and GCM completion, then rebuilt raw/SWA and annual evaluations. Keep cosine results explicitly historical and preserve the separate horizon-training hold. |
| r006 | [Do the principal and spectral-adjusted corrections preserve extreme-event skill?](directions/r006-extreme-event-evaluation/overview.md) | evaluation | active | Nov7 handoff inspection: completed report for 122 starts/40 leads is present and its checksum matches the completion record; scientific interpretation remains pending. | Lianghong: review occurrence/intensity scores and paired intervals, then prepare figures and interpretation using the existing outputs. |

Earlier records: [experiment index](../current_experiments.md),
[2023 memory evaluation](../docs/experiments/v24_2023_stride3_memory_2026-10-04.md).

November 12 corrected cross-cluster handoff: [current read-only portable instructions for Lianghong](../docs/experiments/README_LIANGHONG_PORTABLE_RUNS.md)
cover M1-matched GC–Mamba and four fresh full-GC horizons. Full GC now uses
constant3e-7/no-warmup atlambda0; the new package is
`artifacts/handoffs/lianghong_weather_20261112_constant_lr3em7/`.
Validation passed34 CPU tests and four real full-GC configurations while
preserving672 original computational source files. Original cosine archives
remain immutable and superseded for full-GC execution. No remote job or new
scientific result is claimed.

Code-only delivery is now available in
`artifacts/handoffs/lianghong_weather_20261112_constant_lr3em7_code_only/`:
send its2.28MB code archive and matching checksum. The included README and
download metadata point to the existing495.19MB weights archive on Della,
with an SSH/rsync command and pinned SHA256. Source, templates and launchers
are unchanged; remote SSH/download was not tested. This is a transfer option,
not a new experiment or a public download endpoint.

The dated preparation/ownership records below are historical. Their scheduler
statements describe those earlier checkpoints in the experiment history;
the current schedule and active production IDs are above and in e014.

Prepared 2026-11-09: [full-GC K8/K12/K16/K20 horizons, checkpoint spectra and
SWA](../docs/experiments/v24_full_gc_ar_horizons_2026-11-09.md). K20 update500
reduced development loss4.6434% versus public GC. At the user's request, new
submission is on hold while the reviewer-comparison gaps are addressed: the
draft omits M1 and matched2023 evaluation. Existing M1 results can be reused;
the existing K20 continuation is unchanged.

Reviewer-response ownership (2026-11-07): [Lianghong's README](../docs/experiments/README_LH.md)
assigns full-GC fine-tuning, extreme-event analysis and seed replication to
Lianghong, with the spectral/λ/smoothing study owned by Ilya. Maps/gradient
diagnostics remain unassigned. Alternative PEFT methods (such as LoRA) and
matched GRU/LSTM or fixed-history comparisons also remain unassigned, both rated
2/5 importance and lower priority than the main comparison.
The November7 framing revision removes the equal-budget requirement from
full-GC fine-tuning: that item asks whether further adaptation of the already
fine-tuned baseline helps on the small training dataset. The equal-update-budget
comparison with GC–Mamba moves to the unassigned LoRA item. Completed500-update
full-GC calibrations showed no development improvement; broader conclusions await
the remaining runs. Existing submitted schedules and endpoints are unchanged.
Scope clarification: Lianghong's item2 uses the standard GC loss (λ=0); Ilya's
item1 owns the full five-λ sweep for both methods and smoothing, reusing the
standard-loss GC result as its λ=0 reference. Item1 remains first in the work
order, with importance4/5 and moderate difficulty given the existing pipeline.
The proposed first seed scope is M1 plus matched
reset at seeds23/24; the fresh GC/GC–Mamba pair repeats in the broader plan are a
separate question. No new jobs were launched while documenting this handoff.

Historical initial launch (2026-11-07): [fresh GC–Mamba versus full-GC scalar λ sweep and smoothing](../docs/experiments/v24_fresh_gc_mamba_gc_lambda_smoothing_plan_2026-11-06.md).
Implementation is complete; initial training array 17022621, independent GC/Mamba
controllers 17022622/17022623, and GC/M1 smoothing 17022624→17022625 are submitted
and initially queued. They automatically continue through development LR
calibration, five fresh λ arms per method, filter fitting and common annual
evaluation. Later November7 checks established full-GC feasibility on one80GB
GPU. At user request, pending λ0 job17049162_0 now starts fresh at peak3e-7;
continuation17049163_0 is retained. Other GCFT λ arms remain1e-5, so this revision
is not a fixed-LR λ-only comparison. Original500-update calibration results
(dev32 exact loss increases13.8958%/52.2477% at1e-5/3e-5) are preserved.
New step500 evaluation17062379 supplies matching evidence for the existing
history evaluation. The lower-rate production run has no new skill result yet.
See [r005 e008](directions/r005-mgda-accuracy-spectral-adjustment/experiments.md#e008);
existing MGDA and extreme-event campaigns retain their separate scope/status.
