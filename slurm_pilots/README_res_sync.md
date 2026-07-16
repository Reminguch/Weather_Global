# res=1 / res=2 synchronization SLURM sets

Clean, matched SLURM trios for cross-code synchronization (Lianghong ↔ Ilya).
Two resolutions, identical structure:

- [`res1/`](res1/) — res=1 (1.0°, mesh 5, GraphCast-small frozen baseline, 16-msg)
- [`res2/`](res2/) — res=2 (2.0°, mesh 4, GCv1-K3 frozen baseline, 6-msg)

Each folder has three matched scripts (+ a regression-check + its own README):

| script | mode | key flag |
|--------|------|----------|
| `*_open_loop.slurm`   | open-loop training   | `--feedback-mode baseline` |
| `*_closed_loop.slurm` | closed-loop training | `--feedback-mode closed_loop_sg` |
| `*_swa.slurm`         | SWA build + eval     | post-processes the closed-loop run |

## Key facts
- **One training code for every K.** K is only the `--ar-tail-K` value, passed via the
  `K` env var (`sbatch --export=ALL,K=18 res1_closed_loop.slurm`). A K-scan is the same
  script swept over K. Output dirs are K-tagged so runs never clash.
- **open vs closed are byte-identical except `--feedback-mode`** (and the output dir),
  so running both and diffing the loss trajectory isolates exactly the feedback effect.
- All scripts run in the frozen-clean tree via
  `scripts/training/full_mamba_v20/train_mz_v20.py`. Absolute paths inside — renaming or
  moving the `.slurm` files does not affect execution.

## Validation loss
The trainer logs **training loss only**; there is **no val loss inside the training
loop**. Validation / skill is measured **afterward** by `eval_v22_clean.py` (see each
`*_swa.slurm` step 2): cold_full rollout, 32 inits of 2022, vs ERA5 on the common grid.

## Data-pipeline reproduction (verified)
Re-running these scripts in the frozen tree reproduces the original training loss up to
bf16/GPU nondeterminism:
- **res=1**: resuming the original K20 run from its step-14000 ckpt gives step-14001 loss
  within **0.045%**; two independent bf16 runs of this code differ from each other by the
  same amount (~7e-3) as they differ from the original → the residual is pure bf16/GPU
  nondeterminism, not a data mismatch.
- **res=2**: a fresh K18 run matches the original K18 train_log step 1 within **0.071%**.

(A single-step loss involves a ~24-step AR rollout of a 16-/6-msg GraphCast + Mamba, so
in bf16 the reproducibility floor is ~1e-3, not fp32's ~1e-7.)
