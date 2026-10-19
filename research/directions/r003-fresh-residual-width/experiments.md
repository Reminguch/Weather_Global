# Attempts and evidence

<a id="e001"></a>
## e001 — Seven-year fresh residual width comparison

Event date:2026-10-15
Recorded:2026-10-15 (America/New_York)
Execution: planned
Outcome: pending

**Request:** Design an experiment for residual widths128 and512 after clarifying
that the recent width512 branch used pretrained GraphCast spatial weights.
This request authorizes design; no jobs were submitted or modified.

**Proposed setup:** The [protocol](overview.md) uses2015–2021 train,2022 selection,
2023 evaluation, fresh residual spatial/Mamba initialization, di64/BCG1/rank32,
20k cached open-loop K20 updates and2k closed-loop K20 updates. Fixing BCgroups
and explicit dt_rank removes differences in the historical width comparison.
Open-loop warmup is zero; closed-loop warmup is200 updates.

**Inspected evidence:**

- [Width128 di64 open-loop config](../../../artifacts/checkpoints/v24_Ilya/res1_cached_w128_di_sweep_recovery_20261007/di64_carry/runs/cached/run_config.json)
  specifies fresh residual initialization, width128, di64/BCG1/rank32,
  baseline feedback,K20,20k updates and LR1e-4 with zero warmup.
- [Initialization audit](../../../artifacts/checkpoints/v24_Ilya/res1_cached_w128_di_sweep_recovery_20261007/di64_carry/reports/initialization_identity.json)
  reports0 copied and140 fresh residual parameter tensors. The pretrained
  baseline is separately frozen.
- [Width128 final report](../../../artifacts/checkpoints/v24_Ilya/res1_cached_w128_di_sweep_recovery_20261007/di64_carry/reports/final_report.json)
  verifies20k updates and all required validations. Cached2022 improvement
  was about0.79% at4k, peaked at1.95% at14k, and was1.44% at20k. This motivates
  a20k search budget rather than assuming4k is sufficient for fresh training.
- [Width128 continuation manifest](../../../artifacts/checkpoints/v24_Ilya/res1_cached_w128_closed_loop_k20_20261013/manifest.json)
  identifies di64's14k parent and2k closed-loop continuation. Its final
  [2022 result](../../../artifacts/checkpoints/v24_Ilya/res1_cached_w128_closed_loop_k20_20261013/results/summary.csv)
  is10.7846% pooled exact loss reduction and16.8484% at day10 versus the
  campaign's common GraphCast baseline on32 starts/40 leads.
- [Recent width512 lineage](../../../artifacts/checkpoints/v24_Ilya/res1_cached_w512_closed_loop_k20_20261014/manifest.json)
  instead used baseline_overlay,BCG2,dt_rank=auto and a4k open-loop parent.
  Its14.2078% final2022 loss reduction does not isolate residual width.
- [Cache readiness](../../../data/graphcast/graphcast/dataset/precomputed_baselines/v24_res1_gcsmall_bptt24_k20/READY.json)
  shows complete8-shard,484-chunk coverage. The source2022 validation split
  is separate from the seven training years.
- Read-only Slurm accounting on2026-10-15:15295330 completed20k in37:12:05;
  15295411 completed4k in12:34:07;15575009 completed2k closed-loop in06:39:47.
  Batch MaxRSS values were327827836K,419426104K and259999700K, respectively.

**Provenance:** Planning from saved configs, immutable campaign manifests,
initialization audits, results and Della accounting. The working checkout is
dirty with unrelated work; no new executable snapshot or training configuration
has been frozen for this proposal. Historical sources identify each old run;
they do not identify a new paired runtime yet.

**Observation:** New paired results pending. Existing width128 reuse is
conditional on a full compatibility audit. The2023 benchmark has prior project
exposure, so this design does not label it pristine confirmation.

**Next action:** Follow the corrected data-scaling proposal below. This
width-only proposal will not be executed as a separate experiment.

### 2026-10-15 — User clarified the scientific question

The intended question is whether larger training data helps. The previous
recommendation to defer that comparison is superseded. This proposal was never
submitted and has no result; the direction is closed. The corrected proposal
is [r004/e001](../r004-cached-training-data-scale/experiments.md#e001).
