# res=2 synchronization trio (open / closed / SWA)

Matched res=2 (2.0°, mesh 4, width 512, **GCv1-K3 frozen baseline**, 6-msg-step
baseline) SLURM scripts, mirroring `slurm_pilots/res1/`. All run in the frozen-clean
tree via `scripts/training/full_mamba_v20/train_mz_v20.py`.

**One training code for every K** — K is just `--ar-tail-K` (env var `K`, default 18).

res=2 vs res=1 differences: `--resolution 2.0 --mesh-size 4`, `--baseline-msg-steps 6`
(res=2 GCv1 baseline, not 16), GCv1-K3 baseline ckpt (`results/GCv1/K3_cascade_res2_m4_w512_mp6/`),
`--warmup-steps 200`, `--max-steps 2000 --checkpoint-every 500`, res=2 prepared stream
(`iv9432/.../prepared_stream/res2`) + `v22_res2_GCv1K3_synthetic` residual root.

| file | what | key flag |
|------|------|----------|
| `res2_open_loop.slurm`   | open-loop training  | `--feedback-mode baseline` |
| `res2_closed_loop.slurm` | closed-loop training| `--feedback-mode closed_loop_sg` |
| `res2_swa.slurm`         | SWA build(1k–2k) + eval | post-processes the closed-loop run |
| `res2_regression_check.slurm` | (verification) K18 fresh 1..10 vs original | — |

## Run
    sbatch res2_open_loop.slurm                       # default K=18
    sbatch --export=ALL,K=10 res2_closed_loop.slurm   # a specific K
    for K in 2 4 8 12 16 18 20 22; do sbatch --export=ALL,K=$K res2_closed_loop.slurm; done
    sbatch res2_swa.slurm                             # after closed reaches step 2000

## Eval / val loss
Same as res=1: no val loss in the training loop; measured afterward by
`eval_v22_clean.py` (res2_swa.slurm step 2), cold_full, 32 inits 2022, `--resolution 2.0`
(the 1.5°-levels zarr is downsampled internally). Renaming/moving `.slurm` files does
not affect execution (absolute paths inside).
