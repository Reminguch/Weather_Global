# v23 — d_conv ablation on top of v22 (d_conv: 4 → 8)

**Single-knob ablation.** Same architecture, data pipeline, and training recipe as v20/v22, with one change:

```
temporal_d_conv: 4 → 8
```

This doubles the causal-conv receptive field inside each Mamba block from 4 AR steps (24h) to 8 AR steps (48h). Everything else identical: hidden_size=128, d_state=16, dt_rank=auto(=32), layers=2, location=mesh_processor_interleaved.

## No new training script

Because `train_mz_v20.py` already exposes `--temporal-d-conv` as a CLI argument, v23 does **NOT** need its own `train_mz_v23.py`. The K=1 spinup and K=22 production runs both reuse `scripts/training/full_mamba_v20/train_mz_v20.py` with the new flag.

Launchers live in `slurm_pilots/`:
- `slurm_pilots/v23_K1.slurm` — K=1 spinup (3000 steps, resume from v15 step20000)
- `slurm_pilots/v23_K22.slurm` — K=22 production (3000 steps, resume from v23 K=1 step23000). **Has built-in ckpt-existence guard**; will refuse to launch if K=1 hasn't finished.

Post-K=1 sanity check:
- `scripts/training/full_mamba_v23/check_v23_K1.py` — run AFTER K=1 finishes, BEFORE submitting K=22. Verifies ckpt exists, run_config has `temporal_d_conv=8`, conv1d kernel shape is `(128, 8)`, loss trajectory is sane, and initial loss is within 20% of v22 K=1.

## Launch sequence

```bash
# 1. Submit K=1 (~5-6h on 1× A100 gpu80)
sbatch slurm_pilots/v23_K1.slurm

# 2. After K=1 finishes — verify before launching K=22
python scripts/training/full_mamba_v23/check_v23_K1.py
# Should print "PASS — safe to sbatch slurm_pilots/v23_K22.slurm"

# 3. Submit K=22 production (~5-6h on 1× A100 gpu80)
sbatch slurm_pilots/v23_K22.slurm
```

## Expected outputs

```
results/v23/
  K1_from_v15v2_20k/
    run_config.json                # will show "temporal_d_conv": 8
    v13_residual_step{20500,21000,…,23000}.pkl
    train_log.json
  K22_from_v23K1_23k/              # created after K=22 launches
    v13_residual_step{23500,…,26000}.pkl
    train_log.json
```

## Expected param count

v23 will have **~2,048 more params** than v22 (= 4 × 128 × 4 extra conv-kernel slots across 4 Mamba blocks). Total residual params ≈ 11,092,367 (vs v22 = 11,090,319).

## Expected memory impact

+~1 GB activation/state retention vs v22 K=22. Should fit easily on 80GB A100 (v22 K=22 still has ≥10GB headroom).

## Ckpt warm-start

**Correction (2026-05-23, after a failed first launch)**: the v15 ckpt **does** contain Mamba weights — it was trained with d_conv=4 (4 conv1d kernels shape (128,4), plus a residual_state with conv_cache shape (1,10242,128,3)). Loading it directly into a d_conv=8 model fails with a conv_cache shape mismatch (the symptom we saw in the first v23 K=1 attempt, job 8641313).

Fix: `scripts/training/full_mamba_v23/strip_state_v15.py` produces a v23-compatible warm-start by:
1. Front-zero-padding each `mamba_block_*/conv1d/kernel` from `(128, 4)` to `(128, 8)`. v15's learned weights occupy positions [4:8] (the "recent past"); the new positions [0:4] start at zero. This preserves v15's effective behavior at init.
2. Dropping `residual_state` entirely. Mamba state is fresh-initialized at training start anyway (train_mz_v20.py zeros it), so we lose nothing.

The processed ckpt lives at:
```
/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v23_helpers/v15_step20000_for_dconv8.pkl
```

v23 K=1 slurm now resumes from this processed file. The script can be re-run any time:
```bash
python scripts/training/full_mamba_v23/strip_state_v15.py
```

## Comparison targets

| Run | d_conv | K | Total MSE-imp @ 240h |
|---|---|---|---|
| v20 K=14 | 4 | 14 | +3.56% |
| v22 K=22 | 4 | 22 | +12.69% |
| v23 K=22 (target) | **8** | 22 | TBD — does longer conv RF help long-lead? |

If v23 K=22 > v22 K=22 → local temporal receptive field matters. If similar → d_conv=4 already saturates.

## Compiler note (XLA Triton GEMM disabled)

v23 chain segments that **resume from a prior ckpt** trigger XLA to select Triton nested-GEMM fusion (`__triton_nested_gemm_fusion`), which peaks at ~36 GiB scratch buffer during compilation. This exceeds the effective contiguous memory available on cluster A100 80GB nodes after driver/system overhead — observed in jobs 8691699 (della-l02g1), 8694234 (della-l02g6), 8702678 (della-l02g11), all OOMing at the same 35.96–36.00 GiB single allocation.

**Fresh-init runs (smoke + chain seg 1) did NOT trigger this**: XLA traced a slightly different graph and stayed on cuBLAS GEMM.

**Mitigation applied to chain slurm**: `export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"`. This forces cuBLAS GEMM (small fixed scratch). Cost: ~5–10% slower training. Does not change architecture, loss, optimizer, or numerical model semantics — only the compute kernel that materializes the same forward/backward graph. Floating-point reduction order may differ slightly (bf16 GPU nondeterminism), but this is within normal training noise.

`v22` historical results are **unaffected** (different `d_conv=4` did not trigger Triton in the first place; the env var is set only in v23 slurms).

## See also

- Architecture reference: `/scratch/gpfs/DABANIN/lm8598/Weather_Global/docs/model_source_snapshots/v20_v22_mamba_2026-05-23/README.md`
- v22 result writeup: `results/2026-05-23-v22/README.md`
- v20 result writeup: `results/2026-05-20-v20/README.md`
