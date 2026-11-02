# Research

Updated: 2026-10-28 (America/New_York)

This index starts with the current matched memory-training comparison. Earlier
experiments remain in the existing records; this is not a historical migration.

| ID | Direction / question | Category | Status | Current conclusion | Next step |
| --- | --- | --- | --- | --- | --- |
| r001 | [Does training with memory improve the selected configuration?](directions/r001-matched-memory-training/overview.md) | training | active | Matched result incorporated into arXiv text and Figure 6: carry14.424% vs reset-trained9.163% rollout reduction; paired advantage5.261pp [4.698,5.862], robust to prescribed sensitivities. | Replacement complete; training-seed variability remains unmeasured. Replication is needed only for broader claims across training realizations. |
| r002 | [How does the selected correction compare with higher-resolution GraphCast?](directions/r002-common-grid-forecast-comparison/overview.md) | evaluation | active | Approved ENS-relative Figures 5/9 incorporated (e010); official GC025 checksum verification passed (e011). Figure 2a pilot passed 480 score checks: days 1/2/3/5 within 0.8% of digitized paper, day 10 −4.45% (e012). Full 730-start run active on existing native 2018 data. | Inspect annual array15870083, report15870088 and plot15870456 before claiming paper parity. Archive provenance and native-loss repair remain separate. |
| r003 | [Fresh residual width comparison](directions/r003-fresh-residual-width/overview.md) | training | closed | Unexecuted proposal superseded: the user clarified that the question is the benefit of larger data. | Follow r004. |
| r004 | [Does more cached training data improve each residual width?](directions/r004-cached-training-data-scale/overview.md) | data | active | Oct28:20y replacement submitted; verification16392462 passed, prepare16392486 queued; new-arm startup16392467/68 waits on cache. Both7y controls reused; obsolete pending40y chain cancelled. No data-size result. | Inspect new2002–2022 preparation/cache progression, then two20k-open/2k-closed lineages and common2023 evaluation. |
| r005 | [Can MGDA improve spectral fidelity while preserving exact GC accuracy?](directions/r005-mgda-accuracy-spectral-adjustment/overview.md) | training | active | Nov2 inspection: e004 completed Oct29, all three200-update arms and evaluation. MGDA exact loss+0.4275% vs parent while secondary power error44.5751%→43.2507%; no demonstrated joint win. | Review the spectral/accuracy tradeoff on this small single-seed development cohort; no automatic extension or resubmission. |

Earlier records: [experiment index](../current_experiments.md),
[2023 memory evaluation](../docs/experiments/v24_2023_stride3_memory_2026-10-04.md).
