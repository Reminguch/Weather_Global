# Current experiments

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
