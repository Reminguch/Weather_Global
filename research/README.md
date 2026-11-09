# Research

Updated: 2026-11-09 (America/New_York)

This index starts with the current matched memory-training comparison. Earlier
experiments remain in the existing records; this is not a historical migration.

| ID | Direction / question | Category | Status | Current conclusion | Next step |
| --- | --- | --- | --- | --- | --- |
| r001 | [Does training with memory improve the selected configuration?](directions/r001-matched-memory-training/overview.md) | training | active | Matched result incorporated into arXiv text and Figure 6: carry14.424% vs reset-trained9.163% rollout reduction; paired advantage5.261pp [4.698,5.862], robust to prescribed sensitivities. | Replacement complete; training-seed variability remains unmeasured. Replication is needed only for broader claims across training realizations. |
| r002 | [How does the selected correction compare with higher-resolution GraphCast?](directions/r002-common-grid-forecast-comparison/overview.md) | evaluation | active | Approved ENS-relative Figures 5/9 incorporated (e010); official GC025 checksum verification passed (e011). Figure 2a pilot passed 480 score checks: days 1/2/3/5 within 0.8% of digitized paper, day 10 −4.45% (e012). Full 730-start run active on existing native 2018 data. | Inspect annual array15870083, report15870088 and plot15870456 before claiming paper parity. Archive provenance and native-loss repair remain separate. |
| r003 | [Fresh residual width comparison](directions/r003-fresh-residual-width/overview.md) | training | closed | Unexecuted proposal superseded: the user clarified that the question is the benefit of larger data. | Follow r004. |
| r004 | [Does more cached training data improve each residual width?](directions/r004-cached-training-data-scale/overview.md) | data | active | Oct28:20y replacement submitted; verification16392462 passed, prepare16392486 queued; new-arm startup16392467/68 waits on cache. Both7y controls reused; obsolete pending40y chain cancelled. No data-size result. | Inspect new2002–2022 preparation/cache progression, then two20k-open/2k-closed lineages and common2023 evaluation. |
| r005 | [Can MGDA improve spectral fidelity while preserving exact GC accuracy?](directions/r005-mgda-accuracy-spectral-adjustment/overview.md) | training | active | Nov9 GC/M1 smoothing analysis: raw M1 beats filtered GC on loss and macro power error, but raw GC has the lowest power error. Raw GC/M1/band-GC power errors59.47/67.57/75.70%; M1 and band-GC loss reductions14.38/9.39%. | Integrate the completed reference spectra into the fresh λ-model comparison when its evaluations finish; retain the separate MGDA attempt history. |
| r006 | [Do the principal and spectral-adjusted corrections preserve extreme-event skill?](directions/r006-extreme-event-evaluation/overview.md) | evaluation | active | Nov7 handoff inspection: completed report for 122 starts/40 leads is present and its checksum matches the completion record; scientific interpretation remains pending. | Lianghong: review occurrence/intensity scores and paired intervals, then prepare figures and interpretation using the existing outputs. |

Earlier records: [experiment index](../current_experiments.md),
[2023 memory evaluation](../docs/experiments/v24_2023_stride3_memory_2026-10-04.md).

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

Launched follow-up (2026-11-07): [fresh GC–Mamba versus full-GC scalar λ sweep and smoothing](../docs/experiments/v24_fresh_gc_mamba_gc_lambda_smoothing_plan_2026-11-06.md).
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
