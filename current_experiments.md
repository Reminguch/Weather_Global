# Current experiments

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
