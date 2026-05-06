# GC-Mamba Dimensions

GC-Mamba keeps GraphCast's latent width separate from Mamba's internal SSM
channel width.

## Shape Map

- `width`: GraphCast latent input/output dimension. This is the feature size
  entering the temporal block and the feature size returned to GraphCast.
- `d_inner`: number of scalar SSM channels after Mamba's input projection.
  The temporal block projects `[width] -> [d_inner]`, runs one scalar SSM per
  channel, then projects back with `out_proj`.
- `d_state`: hidden memory size per scalar SSM channel.

For a mesh-latent batch, the SSM memory has shape:

```text
[batch, mesh_nodes, d_inner, d_state]
```

For each channel `i`:

```text
u_i: scalar input to channel i
h_i: [d_state] memory vector for channel i
y_i: scalar output from channel i
```

All `y_i` values are stacked into a `[d_inner]` vector, and `out_proj` maps
that vector back to `[width]`.

`d_state` is the memory depth inside each SSM channel. It is not the number of
parallel channels. `d_inner` is the channel count.
