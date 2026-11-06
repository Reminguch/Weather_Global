# Loss v2 checkpoint evaluation

This publication snapshot reports the ar_tail_k=20 checkpoint at update 1,900. Every evaluation uses all 15 fixed 2022 validation segments (60 chunks), the same native tail20 inputs, 50–1000 hPa targets, Gaussian latitude weights, pressure weights and train-only 6 h-difference scales. Memory resets per segment and carries across its four chunks.

The total weighted normalized MSE is **3.3667028**, versus **3.5251651** for the official pretrained NeuralGCM zero-head reference: **4.50% lower**. All five scored fields improve at this checkpoint.

| Variable | Baseline contribution | Residual contribution | MSE reduction |
|---|---:|---:|---:|
| Temperature | 0.5466318 | 0.5340307 | 2.31% |
| Geopotential | 0.5522369 | 0.5144718 | 6.84% |
| Zonal wind | 0.7131394 | 0.6882320 | 3.49% |
| Meridional wind | 0.5387227 | 0.5092929 | 5.46% |
| Specific humidity | 1.1744344 | 1.1206754 | 4.58% |

The tail-lead total weighted MSE reductions are:

| Lead | MSE reduction |
|---|---:|
| 6 h | 3.05% |
| 12 h | 0.17% |
| 24 h | 0.26% |
| 48 h | 0.82% |
| 72 h | 3.11% |
| 120 h | 6.04% |
| 126 h | 6.87% |

The complete validation curve generally decreases but is not monotonic. This snapshot uses the latest completed checkpoint, not a validation-selected best checkpoint. Percentages describe normalized weighted MSE, not raw physical RMSE. These 2022 monitoring results do not establish held-out 2023 skill or an isolated causal memory benefit. Cloud fields are input-only and are not scored.

The independent v2 ar_tail_k=1 run has saved its first 100-update checkpoint; its common-tail20 evaluation was submitted as job 16946248 and had not completed at the time this report was prepared. No cross-arm conclusion is available yet.

[Training, evaluation and per-variable plots with CSV data](separate_plots/README.md). The plots are a publication snapshot; continuing jobs update their scratch output automatically.

[Exact checkpoint evaluation receipt](eval_step_01900.json).
