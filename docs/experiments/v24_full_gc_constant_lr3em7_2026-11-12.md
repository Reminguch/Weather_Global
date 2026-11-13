# Full-GC constant learning-rate correction — November 12, 2026

The user requested keeping full-GC fine-tuning at **constant `3e-7`**, after
checking the GraphCast autoregressive fine-tuning schedule. This supersedes the
[earlier positive-lambda cosine restart](v24_full_gc_positive_lambda_lr3em7_restart_2026-11-12.md).
Use `3e-7` from the first optimizer update, with **zero warmup and no decay**.
The preserved configuration validator requires `end_learning_rate: null` for a
constant schedule; this means there is no endpoint override, not a zero LR.

All five lambda arms (0, 0.25, 0.5, 0.75, 1) start fresh from public GC-small
with fresh optimizer state. The earlier lambda0 checkpoint at1250 used cosine
decay and warmup, so it is retained as historical evidence, not reused as a
constant-rate arm. Its existing scores do not measure the new schedule. The
user-directed exclusion of old full-GC `1e-5` lineages remains in force.

The remaining full-GC recipe is unchanged: seed22, batch4, BPTT24/K20,
segment96, full weather-feedback gradients, the existing grid-MSE plus stable
AMSE-excess objective and frozen shuffled order. Train2500 updates, split into
two allocations with exact optimizer/RNG/data continuation at1250. Finite
loss increases, drift and weak metrics remain warnings; preserve scheduled
updates and let valid runs finish and receive evaluation.

## Source and validation

Campaign: `artifacts/checkpoints/v24_Ilya/res1_full_gc_constant_lr3em7_20261112`.
Run IDs: `gcft_s22_l{000,025,050,075,100}_lr3em7_constant_20261112`.
The frozen source preserves all649 original November7 computational files
byte-for-byte and adds two orchestration files. Source digest:
`e8e07f2a7c3cd549e37a6cdc88c967a51fdb549960a2bc0f1df78afc30a70c88`.

Nine focused training tests passed, including actual Optax constant-rate
updates and serialized optimizer continuation through1250→2500. Actual
configuration validation checks the optimizer LR at indices0/99/100/1249/1250/2499.
Ten portable full-GC tests and34 downstream/report tests passed. Independent
review found no training-code blocker. These are CPU/provenance checks; they
do not establish scientific improvement or predict the result of training.

Prior comparable1250-update jobs took19:57:40–20:07:45 with224.534–228.220GiB
RSS. Retain1×80GB GPU,8CPU,384GiB and30h per training allocation, with no
array throttle. No additional GPU qualification or LR-calibration run is needed.

## Replacement and evaluation

**Submitted:** training arrays17429091_0–4→17429092_0–4, history17429093_0–4,
and completion17429094. No new constant-rate training result is available.

| Job | Work | Dependency |
| --- | --- | --- |
| 17429091_0–4 | Fresh full-GC updates0→1250 | None |
| 17429092_0–4 | Exact continuation1250→2500 | aftercorr:17429091 |
| 17429093_0–4 | Development checkpoint histories | aftercorr:17429092 |
| 17429094 | Validate full-GC training completion | afterok:17429092 |
| 17429215 | Materialize raw2500 and SWA comparison | afterok:17429092:17406862 |
| 17429216_2–21 | Twenty forecasts; reuse M1 references | afterok:17429215 |
| 17429217 | Raw/SWA MSE and spherical-spectrum report | afterok:17429216 |
| 17429218 | Launch development fitting and annual smoothing | afterok:17429092:17406862 |

Pre-activation inspection found all affected cosine jobs still pending and
no new positive-lambda cosine checkpoints. Canceled the pending oldlambda0
continuation/history17049163_0/17049164_0, cosine arrays17421838/17421839,
history17421840, finish17421841, mixed final17422018/17422019/17422020 and
annual controller17422049. Preserve all existing weights, logs, manifests and
frozen source. The saved-checkpoint diagnostic17404694_2–5 and17404732 has
completed successfully; preserve its results.

GC–Mamba training/history17406861/17406862/17406863 is unchanged. Its M1
initial500/main2000 schedule remains separate. New mixed raw/SWA and annual
comparisons depend on the new five-arm full-GC continuation and17406862,
with no retained cosine full-GC arm. Raw/SWA comparison uses four2022 starts
and40leads, reuses completed M1 references and averages total steps1000/1500/2000/2500.
Annual smoothing fits on32 development2022 starts and evaluates122 annual2023
starts, retaining raw/Gaussian/band results and paired reporting. Cross-family
training recipes still differ; this change does not isolate architecture effects.

Activation receipts and current job IDs are recorded under
`artifacts/plans/v24_full_gc_constant_lr3em7_20261112/activation/`.
Final queue inspection confirmed all eight new job groups **pending**, correct
dependencies, canceled jobs absent and all protected GC–Mamba arrays present.
See the [activation receipt](../../artifacts/plans/v24_full_gc_constant_lr3em7_20261112/activation/complete.json)
and [active five-arm selection](../../artifacts/plans/v24_full_gc_constant_lr3em7_20261112/active_full_gc_selection.json).
New evaluation roots are
`artifacts/evaluations/v24_constant_lr_mixed_final_comparison_20261112` and
`artifacts/evaluations/v24_constant_lr_gc_lambda_annual_20261112`.
Both frozen plan entry points and all eight Slurm test-only requests passed.
Mixed plan hash: `2812ae853d31e1bc0658e1685f57dcc05277ae2d73cdb1105ad767713947929d`;
source: `c5f4e5df5ec693f6b49584b89073c330da87b8d6c2c58e5c16b7260bb83eba23`.
Annual manifest hash: `007cacecd30ff61814c16be4bef78e7d9dd7c513dc616b76f177aabdb67f193e`;
source: `eedcb395ed30cddfb1cde26f04cd991adffe1cb1baaaf6a65d0099d44939871e`.

## Lianghong handoff

The revised [portable instructions](README_LIANGHONG_PORTABLE_RUNS.md) apply the
same constant3e-7/no-warmup schedule to all four fresh full-GC K8/K12/K16/K20
horizons atlambda0. GC–Mamba templates and computation remain unchanged.
The new package root is
`artifacts/handoffs/lianghong_weather_20261112_constant_lr3em7/`.
The original handoff archives remain immutable and are superseded for full-GC
training. Use the new code archive and a fresh full-GC experiment directory;
never reinterpret an already-trained cosine trajectory as constant-rate.
No LH-side installation, training, submission or message delivery is claimed.

For code-only sharing, use
`artifacts/handoffs/lianghong_weather_20261112_constant_lr3em7_code_only/`.
Its2.28MB code archive includes the same computation plus instructions and
pinned checksums for downloading the existing weights archive from Della.
The matching outer `SHA256SUMS` covers only the code archive. Download requires
an authorized SSH account with project-directory read access; no public URL
or remote download has been established.

All34 packaged CPU tests passed. Real-input preparation validated four fresh
full-GC configurations and their actual constant-rate optimizers. All672
upstream computational files, both GC–Mamba stage templates and its launchers
are identical to the original handoff. New and original archive checksums
passed. [Validation evidence](../../artifacts/handoffs/lianghong_weather_20261112_constant_lr3em7/validation/validation.json)
records the checks and the revised portable source digest
`ddd70bc3ee74e2a1465dc7b6482b12bfcce8e83fa53b43e4038ca83d4e5a3215`.

The GraphCast supplement section4.5 uses constant3e-7 in the final autoregressive
stage, while earlier one-step training uses warmup and cosine decay; matching
that rate does not imply matching its full training protocol. See
[GraphCast supplement](https://arxiv.org/html/2212.12794v2).
