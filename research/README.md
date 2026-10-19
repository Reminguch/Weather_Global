# Research

Updated: 2026-10-18 (America/New_York)

This index starts with the current matched memory-training comparison. Earlier
experiments remain in the existing records; this is not a historical migration.

| ID | Direction / question | Category | Status | Current conclusion | Next step |
| --- | --- | --- | --- | --- | --- |
| r001 | [Does training with memory improve the selected configuration?](directions/r001-matched-memory-training/overview.md) | training | active | Matched result incorporated into arXiv text and Figure 6: carry14.424% vs reset-trained9.163% rollout reduction; paired advantage5.261pp [4.698,5.862], robust to prescribed sensitivities. | Replacement complete; training-seed variability remains unmeasured. Replication is needed only for broader claims across training realizations. |
| r002 | [How does the selected correction compare with higher-resolution GraphCast?](directions/r002-common-grid-forecast-comparison/overview.md) | evaluation | active | PrincipalM1/public-checkpoint2023 previews complete: all122 starts,121--122 paired valid samples per diagnostic/lead; independent pooling and bootstrap checks passed (e008). Day10 reductions7.6785--11.3892% versus publicGC025. | Review new Figure5/9 previews and methods before replacing current68-date K22 assets; reconcile native-loss sensitivity separately. |
| r003 | [Fresh residual width comparison](directions/r003-fresh-residual-width/overview.md) | training | closed | Unexecuted proposal superseded: the user clarified that the question is the benefit of larger data. | Follow r004. |
| r004 | [Does more cached training data improve each residual width?](directions/r004-cached-training-data-scale/overview.md) | data | active | Oct17: width512/7y has completed12k/20k open updates; full source download complete, forty-year preprocessing/cache still queued. No final data-size result. | Finish open training and common closed fine-tuning; forty-year arms await the maintenance-blocked producer and verified cache. |
| r005 | [Can MGDA improve spectral fidelity while preserving exact GC accuracy?](directions/r005-mgda-accuracy-spectral-adjustment/overview.md) | training | active | Qualification15787947 queued; release15787948 conditionally launches three200-update arms. No scientific result yet. | Review numerical qualification, then parent/control comparisons on eight2022 starts. |

Earlier records: [experiment index](../current_experiments.md),
[2023 memory evaluation](../docs/experiments/v24_2023_stride3_memory_2026-10-04.md).
