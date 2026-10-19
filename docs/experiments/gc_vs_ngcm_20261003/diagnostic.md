# 2026-10-03：GraphCast residual 与当前 NeuralGCM residual 的诊断

**October 7 update:** see the [standalone test review](../ngcm_test_review_20261007/REPORT.md)
for completed native K20 and lower-LR runs, positive K20 checkpoint results,
remaining regressions, and the evaluation-watcher failure. The experiment status
below is the historical October 3 snapshot.

**当前证据不能说明 residual 思路本身对 NeuralGCM 无效，但能确认：我们没有复现 GC 成功实验的训练条件。** 差异包括修正所在空间、可学习的历史长度、逐层归一化与目标函数，以及学习率和训练预算。滑窗调权只能调整现有目标之间的比例，无法自动补齐这些差异。

本次直接读取 `Weather_Global/minimalistic_code` 的提交 `4b318ee488cb3820f8819c956f4691aa09e57239`，对照完成的 NGCM strict >30 hPa 实验冻结源码，而非当前工作区中不断变化的文件。历史 v22 的结果与维护中的 v24 实现分开讨论，不把 v24 配置倒推为每个历史 checkpoint 的实际配置。

## 1. 先确认 GC 到底在哪个 K 上 work

从提交中的两个原始 evaluator JSON 重新计算物理 RMSE 改善率，正值表示优于各自 frozen GC baseline。两份 JSON 都记录 **32 个起点、40 个六小时时效**。下表使用各变量在 GC 原生气压层上的汇总值，不是训练 loss。

| 变量 | GC K2，12h | GC K2，120h | GC K22，12h | GC K22，120h | GC K22，240h |
| --- | ---: | ---: | ---: | ---: | ---: |
| 温度 | +1.324% | −0.073% | −0.818% | +1.934% | +5.484% |
| 位势 | +1.308% | +0.047% | −5.189% | +1.415% | +4.547% |
| 纬向风 u | +0.590% | −0.095% | −1.214% | +2.368% | +6.000% |
| 经向风 v | +0.532% | −0.074% | −0.899% | +2.095% | +5.895% |
| 比湿 | +1.594% | −0.696% | −0.568% | +4.264% | +8.438% |

按 ±0.1% 的显示容差，GC K2 在 12h 是 11 个变量中 10 项改善、1 项变差；到 120h 是 3 改善、4 变差、4 近似不变。GC K22 在 12h 是 3 改善、8 变差，但在 120h 和 240h 都是 11 项改善。因此 GC 也有短时与长期 skill 的取舍，不能要求 K2 先在五天上全面改善才能研究长 K。

这些 GC K2/K22 checkpoint 的来源路径均标为 `K*_from_v22K1_23k/...step26000.pkl`。它们不是 fresh 2k 训练的同预算对照。两份 JSON 的 baseline 数组也并非完全相同，而且没有逐起点 ID，不能把 K22−K2 差异全部因果归于 K。这里仅确认两份已保存结果的现象。

证据：[K2 原始数据](evidence/gc/v22_K2_K40.json)、[K22 原始数据](evidence/gc/v22_K22_K40.json)、[重算脚本](code/reproduce.py)、[完整数表](comparison.csv)。GC README 中的“240 anchors”与原始 JSON 的 `n_samples=32` 冲突，本诊断采用原始 JSON。

## 2. 当前 NGCM K2 仍然有独立的问题

strict >30 hPa、自适应组训练 2,000 updates 后，独立的 16 起点评估结果如下。最后一列使用短时选择得到的 best checkpoint（1840），没有根据这 16 起点重新挑选。

| 变量 | step2000，12h | step2000，120h | best1840，120h |
| --- | ---: | ---: | ---: |
| 温度 | −0.271% | −11.276% | −4.860% |
| 位势 | +0.929% | −7.434% | −2.996% |
| 纬向风 u | −0.131% | −3.628% | −1.680% |
| 经向风 v | −0.161% | −2.002% | −1.074% |
| 比湿 | +0.075% | −4.841% | −3.233% |

best1840 的云冰、云液水五天改善率也分别为 −0.831%、−0.226%。**并非只看最后一个坏 checkpoint 才得到“七项变差”；选出的 best 在五天上也全部变差。** 所有这些已完成评估都是有限值，当前主要问题是 forecast skill 退化，不能与早期验证出现 NaN、导致训练进程退出的问题混为一谈。

此前汇报的 12h 位势 +1.88%、比湿 +0.17% 来自 **8 个 selection-validation 起点**；上表来自另 **16 个独立评估起点**，所以数值不同。16 起点不参与这次 checkpoint selection，但其中部分在先前小实验中已被查看，不能称为从未使用过的测试集。

为排除仅因层数不同造成印象偏差，也将 NGCM 限制到 GC 使用的相同 13 个气压层重新汇总：step2000 在 120h 的 T/Z/u/v/q 改善分别为 −17.493%、−9.486%、−4.431%、−2.080%、−4.770%。问题仍在。不过两个模型的分辨率、起点和基线不同，这仍不是直接性能排名实验。

原始结果与四张逐变量训练曲线已在 [strict-mask 报告](../weather_only_20261003/REPORT.md) 和 [旧方案对比](../weather_only_20261003/COMPARISON.md) 发布。

## 3. 哪些实现差异已经确认

| 项目 | GC 成功参考配置 | 当前 NGCM K2 |
| --- | --- | --- |
| 修正位置 | 对 GC 预测的天气场直接加 residual | 修正 NeuralGCM native prognostic state，再经过 decoder |
| 分支输入 | 两帧天气场及 forcings，独立 residual 图网络 | 当前一帧 native 状态及 known features |
| memory 训练 | v24：BPTT 24，96-step segment 内 carry，AR tail 20 | 每个 origin 的两步 memory BPTT，下一 origin 清零 |
| Loss 尺度 | 逐变量、逐层的 `diffs_stddev_by_level` | 训练集 24h 差分统计；除比湿外跨层合并尺度；native head 的输出增量使用另一套统计 |
| 天气场 loss | 归一化格点 MSE，纬度面积权重、气压权重 `p/mean(p)` | >30 hPa：20×谱系数误差 + 0.1×谱振幅误差 + 2×batch/time bias；含时效降权 |
| 层范围 | GC-small 的 13 层，50–1000 hPa | 原始 37 层；当前严格截去 ≤30 hPa，剩余 29 层 |
| 变量 | 6 个高空变量 + 5 个地表变量 | T/Z/u/v/q/云冰/云液水，共 7 个 |
| 优化器 | 参考 v24：AdamW，peak 1e-4，warmup200，cosine 至 1e-5，β₂=.98，clip1，WD1e-4 | Adam，peak2e-3，warmup2000，β₂=.95，eps1e-6，无 clip/WD |
| 预算 | 参考 v24：10k updates，batch1，每 update 24 个预测时刻 | 2k updates，batch2，每 update 4 个预测时刻 |

GC 的 `ar_tail_k` 与 NGCM 的 `K` 不是同一接口参数。维护中的 GC 参考配置还包括 4 个 truth-prefix 时刻，并对 24 个时刻都计 loss；memory 在 segment 的 4 个 chunk 间携带，梯度在 update 边界截断。NGCM K2 的 memory 则只见到两步连续轨迹。GC 参考共计 240k 个监督预测时刻，NGCM K2 为 8k；这是原始计数，不代表样本相互独立或等价的数据覆盖。

谱表示本身不是错误：在相应正交变换和权重下，未滤波的谱系数平方误差可对应面积加权格点 MSE。这里需要对照的是逐层尺度、气压权重、滤波、时效权重及额外 spectrum/bias 项的组合，而非简单地把“谱空间”视为失败原因。

来源：[GC 参考配置](https://github.com/Reminguch/Weather_Global/blob/4b318ee488cb3820f8819c956f4691aa09e57239/configs/experiments/v24_Ilya/res1_optimizer_matrix_20260904/mamba1_di16_bcg1_lr1em4_b1p9_b2p98_cos10k.json)、[DirectResidualNormalizer](https://github.com/Reminguch/Weather_Global/blob/4b318ee488cb3820f8819c956f4691aa09e57239/src/models/graphcast/training/core/model.py#L104)、[GC weighted MSE](https://github.com/Reminguch/Weather_Global/blob/4b318ee488cb3820f8819c956f4691aa09e57239/third_party/graphcast/graphcast/losses.py#L56)、[GC 训练内核](https://github.com/Reminguch/Weather_Global/blob/4b318ee488cb3820f8819c956f4691aa09e57239/src/models/mamba/v24_Ilya/training/endpoint_step.py)、[NGCM 冻结训练内核](evidence/ngcm/trajectory_training.py)、[NGCM 统计](evidence/ngcm/paper_statistics.py)、[strict-mask loss](../weather_only_20261003/code/weather_only/objective.py)。

### 修正空间的影响：有代码依据的机制假设，尚非因果结论

GC 可写为 `y = F(x) + Rθ(x,h)`，目标 correction 为 `truth − stop(F(x))`。NGCM 当前路径为 `z_next = Φ(z) + A Rθ(z,h)`、`y = D(z_next)`，其中 `A` 包括输出尺度和 native increment 约束，`D` 是 decoder。因此 NGCM residual 的监督要经过 decoder Jacobian；例如 Z/u/v 不对应一组同名的自由输出 correction，风需要由涡度/散度转换。

当前 native head 有 193 个输出通道（6×32+1），但约束禁止直接修正 surface pressure，并保留原有 degree-zero 涡度/散度。它们是已记录的稳定性设计，不等于实现 bug。相同范数的参数梯度也不保证产生相同大小、相同方向的天气场改善。**尚未测量 decoder 的条件数或受控的接口 ablation，不能断言 decoder 是唯一根因。** 见 [native 分支](evidence/ngcm/model.py)、[increment 转换](evidence/ngcm/native_state.py)、[约束](evidence/ngcm/corrections.py)。

### 为什么滑窗 rescale 没有保证改善

现有 controller 平衡的是分组参数梯度范数，不是梯度方向、Adam 预条件化后的实际步长，也不是验证集物理 skill。每个 field 的权重在所选层内共用，因此不能替代逐层标准差和气压权重。若两个变量的梯度方向冲突，范数接近仍可能互相损害。当前实验没有测量梯度夹角，不能把“存在冲突”当作已经确认的事实。

全层原始物理 RMSE 曾被高空误差主导，但那不等于训练梯度也始终被高空主导。此前 `beta=1e-5` 与 train-only calibration 组合时，校准系数可能抵消常数缩放，所以不能称为最终 1e5 倍的高空降权。新的 strict-mask 实验已用真实 GPU 梯度验证：≤30 hPa 和所有 native loss 的直接 cotangent 为零，改变这些被排除输出不改变 loss 或参数梯度。它仍然退化，说明**只关高空目标不足以解决问题**。共享参数和物理耦合仍能改变高空状态，这与“高空 loss 为零”不冲突。

## 4. 两个不能再使用的简单解释

1. **不能单凭 `closed_loop_sg` 解释失败。** GC 的成功参考配置也使用这一模式，保留 memory BPTT，截断天气反馈。NGCM 的物理接口不同，截断的具体 Jacobian 不同，但“GC 全链路反传、NGCM 没有”不符合代码。见 [GC target/feedback](https://github.com/Reminguch/Weather_Global/blob/4b318ee488cb3820f8819c956f4691aa09e57239/src/models/mamba/v24_Ilya/training/endpoint_step.py#L307)。
2. **不能只怪 width128 或 fresh 初始化。** GC 的已提交 width128 报告记录 fresh residual 在 10k 时，比 frozen GC 降低 exact original GraphCast rollout loss **6.33%**；SWA2k–8k 为 **3.63%**。这说明窄分支也能 work。该指标是归一化、加权的 rollout MSE，不是上表物理 RMSE；这里核实了报告及来源哈希，未重新执行其外部原始评估。只有两个 eval artifact，也不能证明 eval 已饱和。见 [原报告 PDF](evidence/gc/v24_width128_training_2026-09-14.pdf)、[来源记录](evidence/gc/v24_width128_training_2026-09-14.sources.json)。

较强的 GC width512 参考还使用了 pretrained spatial overlay，而 fresh width128 不使用；宽度与初始化不能单独归因。维护中的实际 residual 分支从天气输入建立自己的图网络，不能照抄历史 README 中“直接读取 frozen processor latent”的表述。见 [实际模型构造](https://github.com/Reminguch/Weather_Global/blob/4b318ee488cb3820f8819c956f4691aa09e57239/src/models/mamba/v24_Ilya/model.py)。

## 5. 学习率问题与已经启动的工作

原 NGCM schedule 从 0 线性 warmup 到 2e-3，用满 2,000 updates；衰减从约第 15k 个 schedule step 才开始。因此原 2k 实验**整个预算均在 warmup，没有 decay**。第 2,000 次实际更新使用约 1.999e-3（schedule 计数从 0 开始）。peak 是 GC 参考的 20 倍，但实际更新受梯度尺度和 Adam 状态影响，不能仅凭 peak 比例断言参数更新也大 20 倍。

| 当前实验 | 状态与用途 | 关键边界 |
| --- | --- | --- |
| 真正 K20，2k updates | 数值 smoke `14937571` 与精确 resume smoke `14937572` 通过；正式作业 `14937573` 启动 | fresh seed22，不是 K2 warm-start；保留原优化器，训练覆盖20步并做20步 memory BPTT |
| K2 学习率安排对照，2k updates | smoke `14938454` 已通过，正式作业 `14938455` 已解除测试依赖，等待调度 | 同一冻结数值源码、数据顺序和 loss/controller；仅 peak 改1e-4、warmup改200 |

K20 的 smoke 验证全部 20 步参与 tape、实际参数更新、冻结 backbone、精确零 excluded cotangent、动态权重以及独立进程恢复参数/Adam/RNG/controller。数值 smoke 的相对梯度误差为 3.09e-7。六步 smoke 的 skill 不用作有效性结论。

K2 新对照保留 Adam β₂=.95、eps1e-6、无 clip/WD 和原衰减定义，所以它在 warmup200 后至2k保持1e-4，**不是把 GC optimizer 整套复制过来**。它检验“学习率安排”这一联合因素，不能进一步区分 peak 与 warmup 的单独作用。代码和执行契约单独存放于 [LR control](code/weather_lr_control/README.md)。新 smoke 使用独立 pilot statistics，测试 exact schedule 下的连续/中断续训；生产统计和 checkpoint 分离。

所有新作业在 scratch，H200 路由 ailab / gpu-test，每次一小时、约55分钟存档并续跑，不使用500步切片上限。两个实验都有独立的全变量120h评估 watcher。**正在跑不等于已经得到改善结果。** 最新记录见 [作业状态快照](status.json)、[K20 计划](runs/k20/plan.json)、[LR 计划](runs/lr_control/plan.json)。

## 6. 如何判断下一步，而非继续靠 training loss 下结论

- 首先读取上述两个2k实验的全变量、逐层物理评估，分别比较最后和预先定义的 best，保留 baseline 为候选。LR 对照与旧 K2 使用相同 origin order、calibration 和 selection，开始后核对配置和初始 baseline。
- 若降低 LR 只是把 forecast 拉回零 residual baseline，而无稳定正收益，应称为“减少破坏”，不能称为“学好了”。K20 也按真实物理结果判断，不因 loss 下降或轨迹有限就视作成功。
- 下一项 loss ablation 应独立测试训练集逐变量逐层差分尺度、面积/气压加权物理 MSE，保留现有接口和已选优化器，再与现有谱/bias目标比较。直接输出 residual 与 native residual 的接口 ablation 应另做；一起修改无法定位原因。
- K2→K20 的 curriculum 仍有合理性。当前 K20 是独立 fresh 对照，没有冒充 curriculum。获得更可靠的 K2 checkpoint 后，再单独记录 transfer 的参数、optimizer/memory 重置策略，并做 transfer smoke。

这份诊断确认了实现不对齐和真实退化，没有宣称任何单一根因已经被新的控制实验所证明。复算方法：在完整仓库执行 `python3 docs/experiments/gc_vs_ngcm_20261003/code/reproduce.py`；脚本只读已存 JSON，生成150行结果及其来源 SHA256，不使用 GPU。
