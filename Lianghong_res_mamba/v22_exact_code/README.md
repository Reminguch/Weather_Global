# Lianghong v22 Exact-Code Harness

This harness is for running the Lianghong v22 AR curriculum with the v22
trainer/model source, isolated from the maintained residual-Mamba code in the
active `src/` tree.

The authoritative trainer is:

```text
origin/AR-Training-Lianghong:scripts/training/full_mamba_v20/train_mz_v20.py
```

The authoritative model snapshot is:

```text
/scratch/gpfs/DABANIN/lm8598/Weather_Global/docs/model_source_snapshots/v20_v22_mamba_2026-05-23
```

The harness intentionally fails preflight if the v22 dependency set is
incomplete. It must not silently fall back to current
`src/models/mamba/residual_mamba/training`.

## Commands

Preflight only:

```bash
bash Lianghong_res_mamba/v22_exact_code/preflight.sh
```

Submit the full K=1 -> K=22 chain after preflight and residual-root checks:

```bash
bash Lianghong_res_mamba/v22_exact_code/submit_k1_then_k22.sh
```

Dry-run the Slurm commands:

```bash
DRY_RUN=1 bash Lianghong_res_mamba/v22_exact_code/submit_k1_then_k22.sh
```

## Default Paths

Prepared data:

```text
data/graphcast/graphcast/dataset/prepared_stream/res1
```

Precomputed residual root expected by `train_mz_v20.py`:

```text
data/graphcast/graphcast/dataset/precomputed_residuals/lianghong_v22_iv9432_res1
```

Output root:

```text
artifacts/checkpoints/lianghong_v22_exact_code_iv9432
```

