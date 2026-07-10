# Frozen snapshot of the CLEAN tree (2026-07-09)

This branch (`AR-Training-Lianghong-v22clean-frozen`) captures the exact on-disk
code state of `/home/lm8598/Weather_Global_experiments_v22clean/` on 2026-07-09
— **the code that trained and evaluated every headline v22/v22cl result**:

- res=1 open-loop v22 K-scan (peak +12.4% paper-weighted MSE @240h, K=22)
- res=1 closed-loop v22cl H=128 K-scan + SWA (+19.2% @240h, K=20)
- res=2 closed-loop inner128 / inner1024 K-scans + SWA (+31.5 / +32.0% @240h, K=22)

## Residual-branch conventions in this tree (Design A)

- Residual head reads the **absolute atmospheric state** (same `current_inputs`
  as the baseline GraphCast).
- `DirectResidualNormalizer`: inputs get **standard** normalization (subtract
  `mean_by_level`, divide by `stddev_by_level`); only the training **target**
  (the residual to predict) is normalized by `diffs_stddev_by_level`.
- Closed-loop (`v22cl`) feedback: `x_{t+1} <- x_t + f_GC + f_res`, with joint
  stop-gradient on (Mamba state h, physical input x) at BPTT chunk boundaries.

This is distinct from the main/dev tree's `residual_inputs` architecture
(residual-valued input lanes, tendency-std input normalization). Mixing the two
conventions was the root cause of the 2026-07-09 H=256 incident — see the
sibling branch `AR-Training-Lianghong-H256frozen` for that snapshot.

## Regression test

`slurm_pilots/test_v22cleanfrozen_res2_train.slurm` retrains res=2 inner128
K=18 for 2000 steps (seed 22, same base ckpt/data as the original run) from
this snapshot; the training-loss trajectory and the step-2000 eval should match
`/scratch/.../results/v22_res2_closedloop/K18_res2_inner128/` (原 base128).
