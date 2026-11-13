# Fresh positive-lambda full-GC runs at peak LR 3e-7

**Superseded later on November12:** the user requested constant3e-7 without
warmup or decay. All pending jobs listed below were canceled before the new
positive-lambda cosine training started. The oldlambda0 continuation was also
canceled; its existing weights remain archived. Follow the
[five-arm constant-rate replacement](v24_full_gc_constant_lr3em7_2026-11-12.md).
The remainder of this document records the earlier activation.

Date: 2026-11-12 (America/New_York).
Status: **fresh training submitted**. Arrays `17421838 → 17421839` contain
four tasks each; history `17421840` and completion `17421841` follow.
All were pending at verification. Final raw/SWA comparison
`17422018 → 17422019 → 17422020` and annual-smoothing controller `17422049`
are also submitted and pending with valid dependencies.

The user explicitly chose to replace the four pending positive-lambda full-GC
continuations with fresh runs at peak learning rate `3e-7`. The existing
lambda-zero trajectory already uses that rate and is retained. The rejected
`1e-5` weights are never used as initializers. Their evidence remains archived
under the [comparison exclusion policy](v24_full_gc_comparison_policy_2026-11-12.md).

## Fixed training recipe

New campaign:
`artifacts/checkpoints/v24_Ilya/res1_full_gc_positive_lambda_lr3em7_20261112/`.

| Setting | New positive-lambda full-GC runs |
| --- | --- |
| Lambda | 0.25, 0.5, 0.75, 1 |
| Initializer | Original public GraphCast-small 1-degree/13-level checkpoint |
| Optimizer state | Fresh for each run; no rejected checkpoint reuse |
| Learning rate | Peak `3e-7`, cosine to `3e-8`, 100-update warm-up |
| Total updates | 2,500; fixed schedule across both allocations |
| Allocations | Fresh 0→1,250, then exact resume 1,250→2,500 |
| Shared settings | Seed22, batch4, BPTT24/K20, segment96, full weather gradients |
| Loss | Existing exact-grid MSE plus lambda times stable spherical AMSE excess |
| Data/order | Same original 2015–2021 prepared data and frozen shuffled order |
| Final checkpoints | Raw update2500 and equal-weight SWA of1000/1500/2000/2500 |

Copy each original positive-lambda declared configuration, changing only the
learning rate/end rate and output identity. Preserve the original training
computation byte-for-byte in a new snapshot with orchestration overlays.
Do not change the lambda-zero configuration/checkpoints, GC–Mamba training,
or the separate four-horizon study. There are no extra LR-calibration or
qualification jobs. Finite weak metrics remain warnings and do not stop the
planned endpoint or evaluation.

The retained lambda-zero trajectory is
`res1_fresh_gc_mamba_gc_lambda_20261107/runs/gcft_s22_l000_lr3em7_fresh_20261107`,
with pending continuation `17049163_0` and history `17049164_0`. The independent
M1-matched GC–Mamba training arrays `17406861/17406862` and history `17406863`
remain unchanged, using their own M1 learning-rate schedule.

## Replacement scope and evaluation

Cancel only the four rejected full-GC continuation tasks `17049163_1–4`, their
history tasks `17049164_1–4`, and obsolete dependent controllers/reports.
The old full-GC finish `17049165` and GC–Mamba finish `17406864` both participate
in the old annual join, which hardcodes the rejected full-GC selection. Replace
those controllers with a new external completion/evaluation controller while
preserving their training/history jobs. Replace mixed-final jobs
`17406897/17406898/17406899`, which pin the old full-GC checkpoints.

The new final comparison combines the existing lambda-zero full-GC trajectory,
four fresh low-rate full-GC trajectories and five M1-recipe GC–Mamba trajectories.
Keep actual model identities and source/configuration hashes. Evaluate final
raw and SWA weights on the same four2022 starts/40leads, reusing completed M1
raw/SWA references. Retain development32 fitting and annual122 raw/Gaussian/band
evaluation with the shared exact-GC denominator and paired uncertainty.
Future weights and results are recorded only after actual completion.

The shared annual controller must not wait for canceled finish jobs or read
the old five-arm full-GC selection as the current experiment. The existing
saved-checkpoint diagnostic `17404694/17404732` stays unchanged. Original
frozen sources, sealed reports and high-rate evidence are preserved.

## Resource and scheduler evidence

Inspection of completed full-GC first-part jobs `17049162_0–4` found elapsed
19:57:40–20:07:45 and batch peak host RSS224.534–228.220GiB. The positive arms
used228.024–228.220GiB. Reuse **one80GB GPU, eight CPUs,384GiB host RAM and30h**
for each1,250-update allocation. Four unthrottled tasks per array retain roughly
1.5× walltime and1.68× host-memory headroom against this direct evidence.

Read-only inspection and the precise cancellation/protection sets are in
`artifacts/plans/v24_full_gc_lr3em7_restart_20261112/inspection/`.
All targeted/protected jobs were pending at this inspection; recheck before
activation. No job IDs in this plan are evidence of scientific completion.

## Activated training and provenance

| New job | Work |
| --- | --- |
| `17421838_0–3` | Fresh updates0→1250, lambda0.25/0.5/0.75/1 |
| `17421839_0–3` | Exact resume1250→2500; `aftercorr:17421838` |
| `17421840_0–3` | Development32 history at500/1000/1500/2000; `aftercorr:17421839` |
| `17421841` | Validate four completed endpoints; no obsolete finish hook |

The obsolete positive-lambda training/history tasks and five old shared
controllers listed above were canceled after verifying their pending state.
Post-cancellation verification confirms `17049163_0`, `17049164_0`, GC–Mamba
`17406861/17406862/17406863`, and diagnostic `17404694/17404732` remain pending
with valid dependencies. Exact cancellation and verification receipts are in
`artifacts/plans/v24_full_gc_lr3em7_restart_20261112/activation/`.

New training manifest hash:
`dfb97567b3aed2b02f95d3bde783d06cf2b2a6c7ae45b3e29b7ed2527cc97f4e`.
Source digest:
`eb614ed459e0a6782de26a49afbcc042a9a136a0b8d8d077617b0d98a5f94df7`.
All649 parent snapshot files are byte-identical; only the new driver and
wrapper were added.18 training/scalar tests, actual frozen-runtime validation
of all four configurations, shell syntax and all four Slurm test-only requests
passed.34 focused downstream/final/scalar tests passed. There was no additional
GPU qualification run and no scientific improvement is claimed from submission.

## Submitted downstream replacements

| Job | Work | Dependency |
| --- | --- | --- |
| `17422018` | Validate final weights and build raw/SWA manifest | `17049163_0:17421839:17406862` |
| `17422019_2–21` | Twenty raw/SWA model evaluations; reuse M1 raw/SWA | `17422018` |
| `17422020` | Exact MSE, spectra/power ratios and raw/SWA comparison | `17422019` |
| `17422049` | Validate all ten training endpoints; launch development32 fitting and annual122 smoothing, then join reports | `17049163_0:17421839:17406862` |

Raw/SWA campaign:
`artifacts/evaluations/v24_low_lr_mixed_final_comparison_20261112/`.
Plan hash `91634a77770f349cfcc69287b459709a8aff4f642b54185ae32f2b4eebe96820`;
source digest `c12c9c5a8f7b3099880e41e5ce23ba03ee9820f40c912606d1af833938f4a348`.

Annual campaign:
`artifacts/evaluations/v24_low_lr_gc_lambda_annual_20261112/`.
Manifest hash `92172c7be26f5bf1c70dfc35b877708cb7ad128b4bd00121c62d38348783fb7f`;
source digest `c0a8c32549e67f4a7477e975851af7b7d02dbe327facf1bafed0009f617f2e4b`.

Both frozen plans select the same five low-rate full-GC models and unchanged
five M1-recipe GC–Mamba models. The annual controller records actual initial/final
GC–Mamba proof evidence and advisory initialization differences before launching
smoothing. It never invokes the canceled legacy finish hooks. Later annual
child job IDs will be recorded by the controller; they are not yet assigned.

All four downstream Slurm test-only requests passed. Final queue verification
confirmed all eight new job groups pending, correct dependencies, protected jobs
present and canceled jobs absent. The machine-readable
[active full-GC selection](../../artifacts/plans/v24_full_gc_lr3em7_restart_20261112/active_full_gc_selection.json)
spans the retained and fresh campaigns. The
[activation receipt](../../artifacts/plans/v24_full_gc_lr3em7_restart_20261112/activation/complete.json)
pins all three manifests/plans and the cancellation/queue evidence.

Monitor training and evaluation with:

```bash
squeue -j 17049163,17421838,17421839,17421840,17421841,17406861,17406862,17406863,17422018,17422019,17422020,17422049
```

Submission is not completion. Inspect sealed training/evaluation completion
records before making a new result claim. The previous low-rate lambda-zero
development scores remain available, but there are no new positive-lambda
low-rate scores yet.
