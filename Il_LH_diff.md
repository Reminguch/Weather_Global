# Ilya vs. Lianghong GraphCast changes

This note compares the checked-out Lianghong implementation in
`third_party/graphcast/` with Ilya's copy in
`Ilya_code/third_party/graphcast/`.

## 1. LoRA

**Lianghong:** no LoRA adapters are added to GraphCast's processor MLPs.

**Ilya:** `deep_typed_graph_net.py` adds optional low-rank adapters to the
mesh-GNN processor edge and node MLPs. They are active only when:

- `lora_rank > 0`, and
- `lora_scope == "processor_mlp"`.

For every adapted base linear layer, Ilya adds:

```text
base_linear(x) + (x @ A @ B) * (alpha / rank)
```

`A` is random at initialization and `B` is initialized to zero. Therefore the
LoRA contribution is initially zero, while the pretrained base projection is
unchanged. The base-layer names are kept compatible with a GraphCast
checkpoint, and the LoRA parameters are additional trainable leaves.

## 2. Zero initialization

Lianghong and Ilya both use two zero-initialized projections in residual-Mamba:

1. one inside every Mamba block, and
2. one after `mesh2grid`, at the end of the complete residual branch.

These are ordinary trainable linear projections whose weights and biases start
at zero. The current Lianghong v9/v20/v22 implementation does not use a
separate multiplicative scalar gate.

### Shared Mamba output initialization

In both implementations, a Mamba layer does:

```text
input
  -> LayerNorm -> in_proj -> SSM scan -> out_proj
  -> residual + out_proj_result
```

When temporal-output zero initialization is enabled, `out_proj`'s weight and
bias are initialized to zero. Thus, at initialization:

```text
out_proj_result = 0
Mamba block output = residual + 0 = residual
```

This makes **each inserted Mamba block** initially a no-op on the mesh latent:
Mamba returns its input unchanged. Lianghong passes
`cfg.temporal_zero_init_out` through `self._temporal_zero_init_out`; Ilya's
residual-Mamba trainer enables the same initialization whenever the temporal
backbone is Mamba.

### Shared full residual-branch initialization

Both implementations also apply one zero-initialized linear head after
`mesh2grid`:

```text
mesh processor + Mamba
  -> mesh2grid
  -> zero-initialized residual output head
  -> residual prediction = 0 at initialization
```

The GraphCast-shaped residual branch itself can produce nonzero hidden features;
the final zero head masks its complete prediction at initialization. The frozen
baseline GraphCast is never zeroed. Therefore:

```text
frozen baseline + zero residual prediction = frozen baseline
```

The architectural location is the same, but the code organization and parameter
names differ:

| Implementation | Where the final zero head is defined | Parameter name |
| --- | --- | --- |
| Lianghong | `GCResidualWithZeroHead` subclass in `scripts/training/full_mamba_v9/train_mz_v9.py` | `temporal_residual_head` |
| Ilya | optional path directly in `Ilya_code/third_party/graphcast/graphcast/graphcast.py`, controlled by `_residual_output_head_enabled` | `residual_output_head` |

The two zero points have different purposes:

| Mechanism | Initial effect |
| --- | --- |
| Mamba `out_proj` zero init | Mamba does not perturb the residual branch's mesh features. |
| Final residual output head zero init | The complete residual forecast is zero, so the full forecast starts at the frozen baseline. |

For `gc_mamba`, which inserts Mamba into one GraphCast model rather than adding
a separate correction model, only the first mechanism applies.

## 3. Processor count (`temporal_insert_count`)

**Lianghong:** temporal Mamba is run after every processor step:

```text
processor step 0 -> temporal block r0_s0
processor step 1 -> temporal block r0_s1
...
```

There is one separately named temporal block (and, if stateful, one state) for
each `(processor repetition, processor step)` pair. The training configuration
may contain `temporal_insert_count`, but the Lianghong GraphCast processor loop
does not read it; it does not change this schedule.

**Ilya:** adds `_temporal_processor_group_sizes()` and reads
`_temporal_insert_count`.

- If it is `None`, Ilya uses one group per processor step. This is the same
  processor-then-Mamba schedule as Lianghong, so it can be understood as
  `K = number_of_processor_steps`.
- If it is `K`, Ilya divides the processor steps into `K` near-equal groups.
  For example, 16 steps and `K=3` produce `[6, 5, 5]`.
- In either mode, Ilya runs Mamba **after** the processor step or group:

```text
processor steps 0..5  -> temporal block 0
processor steps 6..10 -> temporal block 1
processor steps 11..15 -> temporal block 2
```

This controls the number of temporal/Mamba insertions and spaces them roughly
evenly. It does not provide arbitrary exact insertion positions; that would
require an explicit list of processor-step indices.

## Files containing the comparison

- LoRA: `Ilya_code/third_party/graphcast/graphcast/deep_typed_graph_net.py`
- Processor count and residual output head:
  `Ilya_code/third_party/graphcast/graphcast/graphcast.py`
- Lianghong's hidden Mamba `out_proj`:
  `src/models/mamba/modules/temporal_mesh_mamba.py`
