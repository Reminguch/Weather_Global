# res=1 synchronization trio (open / closed / SWA)

Matched res=1 (1.0°, mesh 5, width 512, 7yr data, GC-small frozen baseline +
interleaved Mamba residual) SLURM scripts for cross-code comparison. All three run
in the frozen-clean tree (`/home/lm8598/Weather_Global_v22clean_frozen`) via
`scripts/training/full_mamba_v20/train_mz_v20.py`.

**The training code is identical for every K — K is just the `--ar-tail-K` value.**
So the K used is a parameter (`K` env var, default 20), NOT hardcoded. Same script,
any K.

| file | what | key flag |
|------|------|----------|
| `res1_open_loop.slurm`   | open-loop training  | `--feedback-mode baseline` |
| `res1_closed_loop.slurm` | closed-loop training| `--feedback-mode closed_loop_sg` |
| `res1_swa.slurm`         | SWA build(6k–16k) + eval | post-processes the closed-loop run |
| `res1_regression_check.slurm` | (verification) resume orig K20 @step14000, reproduce loss | — |

open vs closed are byte-for-byte identical except `--feedback-mode` and the K-tagged
output dir.

## Run one K (default K=20)
    sbatch res1_open_loop.slurm
    sbatch res1_closed_loop.slurm
    sbatch res1_swa.slurm            # after closed-loop reaches step 16000

## Run a specific K / K-scan
    sbatch --export=ALL,K=18 res1_closed_loop.slurm
    for K in 2 4 8 12 16 20 22; do sbatch --export=ALL,K=$K res1_closed_loop.slurm; done

Output dirs are K-tagged (`results/res1_sync/closed_loop_K${K}/…`) so different K never
clash. For a quick agreement check against another codebase, lower `--max-steps`
(e.g. 1000) and compare the training-loss trajectory.

## Data-pipeline verification (already done)
Resuming the original K20 run from its step-14000 ckpt reproduces the original
train-loss: step 14001 within 0.045%; and two independent bf16 runs of THIS code
differ from each other (~7e-3) by the same amount they differ from the original
(~7e-3) — i.e. the residual is pure bf16/GPU nondeterminism, the data + pipeline
are identical.

## Note on validation loss
The trainer logs **training loss only** — no val loss inside the training loop.
Validation/skill is measured **afterward** by `eval_v22_clean.py` (see `res1_swa.slurm`
step 2): cold_full rollout, 32 inits of 2022, vs ERA5. Renaming/moving these `.slurm`
files does NOT affect execution — all paths inside are absolute.
