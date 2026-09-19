# Current experiments

## Fresh width128 cached di sweep and per-prediction reset — 2026-09-18

- Submitted three gated pipelines: di32 carry **14117930**, di64 carry **14117931**, di16 reset every prediction **14117932**. All were pending at verification. Each initial job requests one full A10040, eight CPUs,64GiB host RAM and6h, account dabanin/QOS gpu-short. No array throttle.
- Fresh residual width128/MP2; Mamba1/BCG1/state16/conv4/rank32/two layers. Preserve the fresh-width128 reference's20k updates,200-update warmup to constant1e-4,batch1,seed22,AdamW(0.9,0.98),weight decay1e-4,clip1,BF16 and FP32 physical/state boundaries. Use cached baseline feedback,BPTT24/K20/segment96,all-step loss. Reset ablation explicitly confirmed by user: reset both SSM and convolution memory before every prediction via temporal_stateful=false.
- Complete existing baseline cache verified against all three configurations. Validate all15 held-out2022 segments every2k updates with exact original GraphCast loss; native checkpoints every1k. These are fresh starts, not continuation from the selected online10k checkpoint. No matched cached di16-carry control or cold10-day evaluation was added to the three requested arms.
- Immutable source digest2cb19c94aa530c7e1bd874d608a2ee13ef799168e778281cf85f4f2e5d51529b. Automatic progression: preflight→numerical parity/20-step replay/benchmarks→70-update resource validation→cached production. Online execution is limited to correctness checks. Production budgets count20k updates and actual validation/checkpoint cadence; trajectories exceeding144h split at native checkpoints with afterok dependencies and exact resume. Correctness checks are never bypassed.
- Fixed the generic initialization gate to validate fresh initialization instead of requiring a pretrained overlay. Added reset-gradient/update equivalence and scheduling regression checks.76 targeted tests passed; shell syntax and all three cache/source/config compatibility checks passed. GPU gates remain pending.
- [Experiment protocol and output paths](artifacts/checkpoints/v24_Ilya/res1_cached_w128_di_sweep_20260918/README.md); jobs.json records initial IDs and per-arm submission/pipeline records track subsequent stages.

## Width128 di16 K20 fine-tuning from selected 10k checkpoint — 2026-09-18

- Submitted training **14117544**, pending at verification; evaluation **14117546**, dependent on successful training. Both request one A10080, eight CPUs and seven hours; training320GiB host RAM, evaluation16GiB. Slurm assigned gpu-short. Completed same-width training13900685 used243.29GiB and37:52 for20k; evaluation13900697 took4:07 for four artifacts.
- Source: fresh-width128 constant-LR experiment's native step10000, best among its exact evaluated checkpoints:10.819934% aggregate GraphCast loss reduction,15.884779% day10. The source already uses K20. Preserve width128/MP2,Mamba1 di16/BCG1/state16/conv4/rank32/two layers,frozen GraphCast Small,width512/MP16,batch1,seed22,2015–2021 training and2022 validation.
- New conservative fine-tuning stage:2000 additional updates,200-update warmup to constant1e-5,AdamW betas(0.9,0.98),weight decay1e-4,clip1. Jointly train all730447 residual parameters. Reset optimizer,RNG,data cursor and initial recurrent state;load trained output heads unchanged. K20/BPTT24/segment96,closed_loop_sg feedback,all24-step loss and recurrent carry/reset schedule remain inherited.
- Added explicit --zero-state-on-init because ordinary batch1 --init-from retains the old checkpoint's nonzero recurrent state while resetting the data cursor. Original source checkpoint remains intact. Tests:57 targeted workspace tests passed;seven first-update/state/CLI tests also passed against the actual frozen source. Native config dry run,shell syntax and all432 source hashes passed.
- Frozen code is cloned from the selected original experiment;only config/runner zero-state support and two experiment launch files differ. The scheduled job requires a three-update warm-start smoke and checkpoint round-trip evaluation before independent production training. Save/check validation every500 updates on the same eight segments;SWA500/1000/1500/2000. Dependent evaluation scores all four checkpoints plus SWA on matched32 cold starts,zero state,40 six-hour leads,exact original GraphCast loss.
- Experiment record: [protocol and outputs](artifacts/checkpoints/v24_Ilya/res1_w128_di16_k20_finetune_20260918/README.md). Exact submission commands in jobs.json;config/source hashes,resource evidence and test reports are saved there. Learning rate1e-5 and2k budget are explicitly recorded new defaults informed by the earlier low-LR pilot,not settings implied by K20.

## Res1 Fourier-spectrum pilot — 2026-09-17

- Corrected to **2023** at the user's request. Replacement GPU array `14080456_0-2` evaluates the same frozen M1/M2/M3 SWAs as the accuracy campaign; report job `14080470` uses `afterok:14080456`. At 23:29 UTC the array was pending for priority. Superseded 2022 jobs `14078754` and `14078766` are cancelled, with partial files preserved.
- Eight fixed 2023 00 UTC starts (January/February/April/May/July/August/October/November 15), staged ARCO ERA5, 40 six-hour leads, cold/full feedback and zero recurrent initialization. Nine physical FP32 fields exported per branch; eleven spectral observables including derived wind speeds. Normalized zonal Fourier power, real cross-power and error power over 30–60 degrees in both hemispheres with cosine-latitude weights; pool powers before ratios and correlation.
- Workflow: `plots/analyze_models/data/resolution_eval/res1_spectral_pilot_2023_20260917/`; figures automatically go to `plots/analyze_models/images/resolution_eval/res1_spectral_pilot_2023_20260917/`. Frozen source/protocol verification, 53 focused update tests and shell checks passed. Each GPU task requests one GPU, eight CPUs, 16 GiB, 1:02; report requests two CPUs, 4 GiB, 15 minutes. Completeness, provenance and canonical M1 baseline checks gate the report. Restored historical GPU numerical settings after the cancelled run's ~118s/lead slowdown; cause and replacement speed remain unverified. Early per-lead timings now logged.
- Details and limitations: [pilot record](docs/experiments/v24_res1_spectral_pilot_2026-09-17.md). This eight-start pilot does not complete the full-year spatial/extreme study; the 365-start 2023 exact-loss campaign remains separate.

## Frozen res1 models, full-year 2023 daily evaluation — 2026-09-17

- Submitted pipeline: GPU preflight `14077246_0-2` (four seasonal 2022 dates per model), validation gate `14077255`, production `14077262_0-68`, then report `14077269`. Each stage depends on successful completion of the preceding stage; no array throttle. Production also checks the saved pilot pass record.
- Frozen M1: batch-4 di16 faster Mamba SWA 500–2000; M2: batch-4 di32 joint LR SWA 500–2000; M3: optimizer-study Mamba1 di16/BCG1 SWA 2000–8000. No selection on 2023 scores. All 365 daily 00 UTC starts from January 1 through December 31, each with two truth inputs and 40 six-hour cold/full-feedback forecasts.
- Workflow: `artifacts/evaluations/v24_res1_2023_daily_20260917/`. `protocol.json` freezes model/config/statistics hashes, UTC starts, code (including vendored GraphCast), package versions, numerical settings, and inference/bootstrap seeds. `submissions.json` records exact commands. Source and protocol verification passed; 57 focused tests passed.
- Existing finalized ARCO ERA5 store provides all required December 2022 / January 2024 padding. Data preflight verified all 1,502 staging records, 16,522 target chunks, eight boundary checksums, and exact agreement of all 11 target fields plus statics at both shared December 2022 timestamps. Seasonal source parity remains unverified; no new download was needed.
- Each GPU task: one A100-80GB, eight CPUs, 16 GiB RAM, 1:02 walltime; 23 shards per model, 15–16 starts per shard. Completed comparable V24 jobs used about 9.5–9.8 GiB, with ~523s first forecast and ~104s steady forecasts. Pilot checks RAM and projected runtime before release. CPU gate/report: two CPUs, 4 GiB, 15 minutes.
- Every start/lead retains exact original GraphCast losses. Final comparisons share M1's canonical GraphCast baseline; other baseline runs must meet frozen relative tolerance 0.001 plus absolute tolerance 1e-6. Deterministic GPU reductions enabled.
- Automatic report: `report/report.md`, `report/loss_summary.json`, summary and per-initialization CSVs, and `report/exact_loss_reduction_2023.png`. Paired 14-day non-wrapping blocks within five calendar segments; 10,000 replicates, seed 20260916; 10-/21-day and annual-block sensitivities. Per-lead pointwise intervals and simultaneous intervals for the three rollout comparisons. This run measures canonical accuracy; spatial/extreme diagnostics remain a separate proposed study.

## res0.25 full24 checkpoint/SWA evaluation after training — 2026-09-15

- Resubmitted on 2026-09-16: repaired evaluation array `14008384_0-23`, followed by merge `14008385` with `afterok:14008384`. Same per-shard GPU/CPU/memory/walltime budget, with no array throttle or obsolete preparation dependency. Verified all 198 source hashes and all six checkpoint hashes before submission. Canceled obsolete merge `13968878`, which was stuck on the failed original array. Submission history is preserved under `repairs/20260916_graphcast_runtime/resubmission_20260916T220840Z/`.
- Repair on 2026-09-16: array `13968877_0-23` failed because the saved source omitted `third_party/graphcast` and imported stock GraphCast without Mamba support (`unexpected keyword argument 'is_training'`). Restored the vendored dependency, added runtime/path checks to preparation/evaluation/merge, and set the snapshot's `PYTHONPATH` in the launcher. All 198 source hashes verified; original files and repair provenance are under `repairs/20260916_graphcast_runtime/`. Fifteen targeted tests passed.
- Saved-snapshot smoke passed on CPU: real BF16 residual initialization and forward pass on a small synthetic grid with two mesh steps, BCG2, two Mamba layers and conv4; 28 temporal parameter modules and eight finite recurrent-state leaves. Script and results are in the repair directory. Full-resolution GPU results are pending the resubmitted array.
- Preparation job `13963227`, dependency `afterany:13659343`: runs when training ends, including timeout. Discovers all atomically saved native checkpoints (100/200/300 currently; 400/500 if saved before termination).
- Build parameter SWA over all saved checkpoints and the last three, deduplicating identical windows. Frozen native builder validates checkpoint schema/configuration/source identities.
- Preparation automatically submits an unthrottled GPU array: four eight-anchor shards per checkpoint/SWA, followed by one exact sufficient-statistics merge. Native di64/BCG2 architecture from run_config; 32 matched cold starts, zero recurrent state, 40 leads on January 2022, seed0, original GraphCast loss and vanilla baseline. RMS-bias diagnostic omitted.
- GPU budget per shard: A10080, 8 CPUs, 128 GiB, 5h. Comparable completed jobs `13160781_2` and `13160787_1` took 3:16–3:17 and peaked at ~61 GiB. CPU preparation: 32 GiB/2 CPUs/1h (3.1 GiB native DP checkpoints, up to five held for SWA). Merge: 16 GiB/2 CPUs/1h.
- Workflow and outputs under `artifacts/checkpoints/v24_Ilya/res0p25_uniform_full24_di64_bcg2_dp4_20260905/legacy_di64_bcg2_uniform_full24_bf16_dp4_lr1em5_sg500/eval/cold_full_zero_exact_gc/post_training_13659343/`. Includes frozen source, source hashes, settings and submission records; completion writes individual JSONs plus `comparison.json` / `comparison.md`.
- Original verification: four tests passed for checkpoint discovery, SWA timeout windows, real evaluator architecture/shard CLI, and dynamic dependency chain; shell syntax and input existence passed. See the repair note above for the subsequent failure and validation.

## v24 cached stepwise versus online, matched res1 4k — 2026-09-14

- Recovery on 2026-09-15: original preflight `13903317` failed after 2:02, before parity/updates, because the module launcher used `python -m __main__`. Fixed the importable worker module name; 19 targeted tests passed, including a real subprocess-launch regression. Replacement preflight `13937054` uses the same 128 GiB / 8 CPU / full A100 40GB / 2h budget with gated automatic progression. New frozen experiment: `artifacts/checkpoints/v24_Ilya/res1_cached_stepwise_pair_20260915/`, source digest `2366a81748b99327bb68f74d3e400d018f08696199533bb781bae91abdcb444b`. The original snapshot/logs are preserved; no 4k training was launched by the failed job.
- Preflight job `13903317`: submitted, pending at verification. One full A100 40GB (`gpu40&nomig`), 8 CPUs, 128 GiB host RAM, 2 hours. Canonical GC-overlay initialization runs in its own process, followed by three updates per backend.
- Both production runs start fresh from identical native step-zero state: width512/MP2, Mamba1 di64/BCG2/state16/conv4, two temporal layers per insertion, batch1, BPTT24, segment96, baseline feedback and all-step loss. AdamW stays at `1e-4` for 4,000 updates, with no warmup or decay, betas(0.9,0.98), weight decay1e-4, clipping1 and seed22.
- Cached execution retains the original one-step model, host recurrent-state tape and reverse-time VJPs. It reads the existing 424 training / 60 validation chunks; producer-hashed files remain unchanged. Common validation reports the original BF16 objective and exact FP32 physical-field GraphCast loss.
- Automatic progression is enabled: measured preflight → full parity and paired20/50-update benchmarks → actual reduced-RAM, four-CPU check → two independent4k runs → dependent checkpoint replay and report. Every transition requires successful, matching provenance reports; failed checks stop advancement.
- The deterministic cache-generation logs show approximately304 seconds per GC chunk, so the older approximately10-second online update log is not used to promise production duration. Full-gate and production allocations are sized from the new measurements.
- Local verification:102 targeted CPU tests passed, including exact small-model online/cached update comparisons, ownership/coordinate restoration, exact validation, checkpoints, scheduling gates and final-report evidence. All90 executed source fingerprints matched the frozen snapshot; production cache metadata validation passed.
- Experiment: `artifacts/checkpoints/v24_Ilya/res1_cached_stepwise_pair_20260914/`. Source digest:`fb155946b252d1cca0ec322190502540c1eb4f0de8a113478e094f3119584434`. `submission_*.json` and `pipeline_status.json` record jobs; `reports/` holds stage evidence and the eventual comparison. The full-resolution gates and both4k trainings are pending.
- Implementation/usage: `docs/experiments/v24_cached_stepwise_training.md`.

## v24 fresh width128, constant LR retry — 2026-09-14

- Training job `13900685`: pending for priority at submission verification. One gpu80 GPU, 8 CPUs, 288G host RAM, 54h, `gpu-medium`. Same-width completed job `13656412_1` took 19:23:09 for 10k updates and peaked at 224.80 GiB host RAM; estimate about 40h for this 20k retry including validation.
- Same fresh width128/MP2 GraphCast/Haiku residual initialization, Mamba1 di16/BCG1, rank32, batch 1, seed 22 and closed-loop training data as the historical narrow run. AdamW warms up over 200 updates, then stays at `1e-4` through update 20,000. Native checkpoints every 1k; fixed validation every 2k on 8 segments.
- Record initial residual parameters and GM/MG, processor, Mamba, correction-head displacement/update norms and estimated post-clipping gradients. Diagnostics-only snapshots at 0/1/10/100/500; exact resume uses native checkpoints. Also build the prespecified late SWA from 16k–20k every 1k.
- Evaluation job `13900697`: pending for dependency `afterok:13900685`; one gpu80 GPU, 8 CPUs, 16G RAM, 6h, `gpu-short`. After training, evaluate checkpoints 5k/10k/15k/20k on the same 32 cold starts and 40 leads using exact original GraphCast loss.
- Experiment: `artifacts/checkpoints/v24_Ilya/res1_fresh_w128_constlr_20260914/`; frozen source/config hashes, config changes, resource evidence, references, and submission commands are saved there. Logs: `logs/v24_r1_fresh128_13900685.{out,err}` and `logs/v24_r1_fresh128_eval_13900697.{out,err}`.
- Pre-submit checks: 8 diagnostics/recovery tests passed, native config dry run and input paths passed, shell syntax and all 430 frozen source hashes verified. Three-update GPU smoke and checkpoint round-trip evaluation run before production and are still pending.
- At 10k, compare schedules at equal update budget; later checkpoints test extra training with constant LR. Periodic validation preserves training RNG/state/cursor.

## v24 fresh width512, constant LR — 2026-09-14

- Training job `13900119`: submitted, pending for priority at launch verification. One gpu80 GPU, 8 CPUs, 288G host RAM, 72h, `gpu-medium`. Comparable width512 batch-1 training took 26:11 for 10k updates with 223.98 GiB host MaxRSS; estimated duration for this run is about 53h.
- Fresh residual width512/MP2 using GraphCast/Haiku spatial defaults, with no pretrained residual overlay. Frozen GraphCast Small baseline; Mamba1 di16/BCG1, batch 1, seed 22; existing zero Mamba output projections and zero correction head retained.
- Train 20,000 updates: 200-update warmup, then constant AdamW LR `1e-4`, no decay. Native checkpoints every 1k; fixed validation every 2k on 8 segments. SWA uses the prespecified 16k–20k checkpoints at 1k intervals.
- Record GM/MG, processor, Mamba, and correction-head parameter displacement, update norms, and estimated post-clipping gradients. Residual-only diagnostic snapshots at steps 0, 1, 10, 100, and 500; metrics also every 1k. Exact resume uses native checkpoints, not these diagnostic snapshots.
- Evaluation job `13900169`: `afterok:13900119`, pending for dependency; one gpu80 GPU, 8 CPUs, 20G host RAM, 4h (scheduler assigned `gpu-short`). Evaluate checkpoints 10k/20k and late SWA on the matched 32 cold starts and 40 leads using exact original GraphCast loss.
- Experiment: `artifacts/checkpoints/v24_Ilya/res1_fresh_w512_constlr_20260914/`; frozen source and config hashes, manifest, resource evidence, and exact submission commands in that directory. Logs: `logs/v24_r1_fresh512_13900119.{out,err}` and `logs/v24_r1_fresh512_eval_13900169.{out,err}`.
- Validation before submission: 14 tests passed, native configuration dry run passed, shell syntax and frozen hashes verified. The job runs a three-update GPU preflight and checkpoint round-trip evaluation before production; these have not run while it is queued.
- Historical pretrained-width512 and fresh-width128 runs used cosine decay over 10k updates, so comparisons also change schedule and training budget.

## v24 batch-4 temporal LR recovery — 2026-09-12

- Fixed the warm-up-to-branch checkpoint path error in `scripts/experiments/run_v24_res1_batch4_lr.py`. Reuse existing warm-ups only after checking the native training checkpoint, completed update count, and matching saved training configuration.
- All three failed di16 arms retain their completed 500-update warm-ups. Replacement training array `13807466_[0-2]` runs joint LR, 0.3x Mamba LR, and frozen spatial weights for 2,000 branch updates plus SWA. Resources: one gpu80 GPU, 8 CPUs, 320G host RAM, 24h per task; measured warm-up MaxRSS was 196–228 GiB and steady updates took about 26.5 seconds.
- Repaired existing evaluations `13617441_[0-2]` to depend on the corresponding replacement training task via `afterok`. Other training/evaluation tasks remain queued; summary `13617458` still depends on both evaluation arrays. No recovery evaluation has completed yet.
- Validation: four driver regression checks and seven existing batch/LR tests passed; all three real warm-ups loaded and matched their configs. Refreshed source snapshot and verified all 28 checksums. Recovery job mapping and evidence are recorded under `artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/jobs.json` and its `checks/path_fix_*` directory.

## v24 res1 baseline cache — 2026-09-07

- GPU pilot: `13570387`; four representative chunks (first/last train and validation), 96 predictions, online parity and exact FP32 disk round-trip checks. Allocation: one gpu80 GPU, 8 CPUs, 32G host RAM, 1:02 walltime.
- Automatic production-submission job: `13570390`, dependency `afterok:13570387`. It requires a successful pilot parity report and completed accounting, sizes memory/time from measured peak and throughput, submits an unthrottled eight-task GPU array, and submits final CPU verification after the whole array succeeds.
- Initial pilot `13570194` generated all four chunks but failed an incorrect bitwise fresh-model-repeat check. It took 4:43, peaked at 16.22 GiB batch MaxRSS, and measured about 5s per steady chunk. Corrected the disk-round-trip check to use original predictions, and enabled deterministic GPU reductions per the existing repository parity-test convention. The `1e-5` normalized online parity tolerance remains unchanged. Failed output is preserved at the pilot path with `_failed_13570194` appended; obsolete release job `13570198` was cancelled.
- Cache: `data/graphcast/graphcast/dataset/precomputed_baselines/v24_res1_gcsmall_bptt24_k20/`; pilot uses the same path with `_pilot` appended.
- Coverage: 424 train chunks (2015–2021), 60 validation chunks (2022), 11,616 predictions, 234.03 GiB FP32 payload. Existing incomplete-segment exclusions: 50 train and 18 validation anchors.
- Schedule: exact submitted training boundaries, 24 steps per chunk, four observed input windows followed by baseline-only AR feedback. No Mamba model/state is constructed by generation.
- Code/usage: `scripts/preprocessing/BASELINE_CACHE.md`; tests: nine passing in `tests/v24_Ilya/test_baseline_cache.py`. Full-data dry run and Python/shell checks passed.
- Logs: `logs/v24_baseline_13570387_4294967294.{out,err}` (Slurm non-array task placeholder); release logs `logs/v24_baseline_release_13570390.{out,err}`. Production IDs/resources will be written to the cache's `submission.json` and appended below; `READY.json` marks successful full verification.
- Budget evidence: res1 evaluation `12530971_3` completed in 40:15 with about 12.04 GiB batch MaxRSS; di64 training preflight `13435249_1` completed in 12:22 with about 20.97 GiB batch MaxRSS. Production uses the new baseline-only pilot's measurements.

## v24 res1 open-loop di64/BCG2 — 2026-09-07

- Training job: `13560724`, submitted with `scripts/experiments/train_v24_Ilya_res1_open_loop_di64_bcg2.slurm`.
- Config: `configs/experiments/v24_Ilya/res1_open_loop_mamba1_di64_bcg2_all24_10k.json`.
- Fresh residual branch using official GraphCast Small spatial-weight initialization; frozen baseline, baseline-only feedback, stateful Mamba1 di64/BCG2, state size 16, two layers, convolution width 4.
- Training: 10k updates, segment 96/BPTT 24/AR tail 20, all-step loss, AdamW LR 1e-4 with betas (0.9, 0.98), 200-update warmup, cosine to 1e-5, seed 22. Checkpoints every 1k; fixed validation every 2k.
- Resources: one gpu80 GPU, 8 CPUs, 288G host RAM, 42h. Evidence: completed res1/di16 jobs `13435248_6/7` took about 26.2h with about 224 GiB batch MaxRSS; di64/BCG4 preflight `13435249_1` measured 10.06s per steady update.
- Output: `artifacts/checkpoints/v24_Ilya/res1_open_loop/mamba1_di64_bcg2_all24_lr1em4_b1p9_b2p98_cos10k_seed22/`.
- Logs: `logs/v24_r1_open_di64_bcg2_13560724.{out,err}`; GPU telemetry uses the same prefix with `_gpu.csv`.
- This initial reference run computes GraphCast online; offline residual caching remains subsequent work.

## v24 res1 optimizer matrix — 2026-09-04

- Goal: stabilize the successful full-loss/SWA recipe while comparing Mamba1 and legacy Haiku initialization.
- Models: `{legacy,mamba1} × {di16/BCG1, di64/BCG4}`; `d_state=16`, two temporal layers, zero output init.
- Optimizers: peak LR `{1e-4,5e-5}` × Adam `(β1,β2)={(0.9,0.999),(0.9,0.98),(0.8,0.98)}`.
- Common training: res1, all 24 loss steps, 10k updates, seed 22, 200-step warmup, cosine decay to 0.1× peak LR, AdamW decay `1e-4`, clip norm 1, checkpoints every 1k.
- SWA windows: 2k–8k, 4k–8k, and 6k–10k.
- Evaluation: screen checkpoints 2k/4k/6k/8k/10k plus all SWAs on 8 anchors; exact matched 32-anchor evaluation for the best checkpoint and SWA from each run.
- Scale: 24 full runs. `di64/BCG4` is released only after two architecture preflights succeed.
- Resources: di16 array uses A100-80GB, 8 CPUs, 224G, 36h; di64 array uses A100-80GB, 8 CPUs, 256G, 48h.
- Selection: exact original GraphCast rollout-loss reduction; also track clipping rate, gradient percentiles, late drift, and SWA gain. Re-run the top two settings with two additional seeds.
- Training jobs: `13435248_[0-11]` di16/BCG1; `13435249_[0-1]` di64/BCG4 preflight; `13435250_[0-11]` di64/BCG4 full matrix (`afterok:13435249`).
- Evaluation jobs: `13435335_[0-11]` di16 and `13435336_[0-11]` di64; each task uses `aftercorr` to screen and exact-evaluate its matching successful training run.
- Requested early-SWA comparison: built six-member SWA(2k,3k,4k,5k,6k,7k) for the ten di16 runs that had reached 7k; exact matched 32-anchor/40-lead evaluation is `13548842_[0-9]`, followed by summary job `13548843`. Outputs go to `results/v24/res1_optimizer_swa2k7/` and compare day-5/day-10 2m-temperature RMSE and exact allvars GraphCast-loss improvement with the incumbent V22 legacy SWA(2k--8k).

## Pending-job audit repair — 2026-09-11

- Cache pilot `13570387` timed out; release `13570390` was permanently blocked and has been replaced. New pilot `13740415`: 40G host RAM, 2h30, fresh `_pilot_audit_20260911` output. New release `13740425` waits for successful pilot parity/accounting and retains the existing conditional production/verification procedure. Old pilot artifacts preserved.
- Width-128 evaluation now records baseline variation and uses a 1% cross-run guard (observed maximum 0.4454% across 38 matched evaluations), while enforcing baseline checkpoint identity and matched evaluation metadata. Exact loss computation unchanged; source snapshot/hash refreshed.
- Audit report: `docs/reports/pending_jobs_audit_2026-09-11/README.md`. No impossible dependencies remain.

- Baseline cache production submission: pilot `13740415` passed; array `13745392` (8 shards), verification `13745393`; 48G host RAM / 660 minutes per shard. Cache: `data/graphcast/graphcast/dataset/precomputed_baselines/v24_res1_gcsmall_bptt24_k20`.
