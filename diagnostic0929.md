# NeuralGCM + Residual Mamba 诊断报告

主要核查日期为 **2026-09-29，美东时间**，**10 月 1 日再次核对了文中列出的产物哈希和 K20 最终失败状态，均未改变**。本文基于实际执行的冻结源码、训练配置、checkpoint、验证结果、历史干预实验和 9 月 29 日 Slurm 查询。最近一批被核查的训练产物写于 9 月 24 日。文中的 step 均指 optimizer update，K 指每个 episode 连续预测的 6 小时步数。

## 1. 当前结论

**现在有三层问题，必须分别处理。**

1. **旧版残差接口存在已被干预实验确认的失稳机制。** 无约束的直接气压修正造成第二步误差陡增，额外注入的零阶散度造成持续气压漂移。旧 loss 又几乎完全由云水项主导。9 月 24 日的后续实现已经限制这两个残差通道，并更换了 loss。
2. **新版 K=1/K=2 的目标下降与主要气象指标改善不一致。** 四组均完成 1,000 步，五项验证 loss 降低约 96%–98%，但 T850、Z500、U850、Q700 等指标明显恶化。高层位势的改善是真实的，它在当前全 37 层目标中占据巨大份额，足以掩盖其他层和变量的损失。
3. **当前 K=2 → K=20 实验已经失败。** 使用 K=2/w128/d16 的第 800 步权重初始化，初始五天 loss 已是原模型的 **238.7 倍**。微调 100 步后达到 **3,113.1 倍**，随后在下一次训练 batch 的 loss/gradient 检查处退出。不能再把它描述为“正在正常训练”。

已有证据支持“目标选择、物理反馈约束和长时训练信号仍不充分”。尚不能仅凭现有 traceback 断言新版第 101 次更新首先坏在某个物理变量、Mamba 层或反向算子。该异常把非有限 loss 和非有限 gradient 合并为同一条消息，需要针对失败 batch 分开复现。

## 2. 实验版本与真实状态

| 版本 | 训练方式 | 实际状态与含义 |
| --- | --- | --- |
| 9/20 `scatter_v3`，9/23 开始训练 | 缓存 K=1，20 epochs，随后计划 K=20 | 三组在第 5 轮后的闭环验证中退出。已完成 2,120/8,480 updates，即 25%。w256/di32 后来被取消，保留到 update 2,600 的 checkpoint |
| 9/24 `feedback_v2` | 加入气压/零阶模式约束，换用池化六小时差分 loss，重新初始化 | 是中间版本，后被新的五项目标实验替代。其 loss 不能与另外两个版本混用 |
| 9/24 五项目标，K=1/K=2 | 在线闭环 episode，w128/w256，均为 d16 | 四组均到达 `MILESTONE_1000.json`，在 2,000 步配置预算的中点按要求停下，没有 `DONE.json` |
| 9/24 五项目标，K=20 | 从 K=2/w128/d16 的最佳权重转移 | job **14387208 FAILED**，最后持久化 update 为 **100**。没有达到 1,000 步 review 或 2,000 步终点 |

9 月 29 日核对到的作业为：

| Job | 任务 | Slurm 状态 | 结束时间，EDT |
| --- | --- | --- | --- |
| 14381988 | K1/w128/d16，1,000 步 | COMPLETED | 9/24 15:58:53 |
| 14381990 | K1/w256/d16，1,000 步 | COMPLETED | 9/24 16:01:46 |
| 14381989 | K2/w128/d16，1,000 步 | COMPLETED | 9/24 16:30:39 |
| 14381991 | K2/w256/d16，1,000 步 | COMPLETED | 9/24 17:09:35 |
| 14386597 | 独立 K2→K20 transfer smoke | COMPLETED | 9/24 17:20:02 |
| 14387455 | 四组短时物理指标评估 | COMPLETED | 9/24 17:28:43 |
| 14387208 | K20/w128/d16 | FAILED，exit 1:0 | 9/24 18:22:07 |

9 月 29 日的 `squeue` 未见这些 NeuralGCM 训练仍在运行。9/24 留存的 `current_experiments.md` 和 `k2_to_k20_w128_v1.json` 中“running”的文字早于最后失败，不能作为当前状态。K20/w256 未在这一轮提交，d32 不属于当前四组短时实验。

## 3. NeuralGCM backbone 与 residual 的具体结构

### 3.1 冻结的 backbone

| 项目 | 实际设置 |
| --- | --- |
| 权重 | 官方 `deterministic_2_8_deg.pkl` |
| 权重 SHA256 | `bdec1b4612c7385fc492aa031db252c66fb74b788a7efbb118b8b60b06644d3e` |
| 库版本 | NeuralGCM 1.2.2，Dinosaur 1.3.3，JAX/JAXlib 0.9.2，Haiku 0.0.16，Optax 0.2.7 |
| 冻结范围 | learned encoder、动力学核心、learned physics、learned decoder 的全部参数 |
| 状态 | 32 个 sigma 层的涡度、散度、温度扰动、比湿、云冰、云液水，以及单层 log surface pressure |
| 时间推进 | 所选 checkpoint 的内部步长为 1 小时；每推进 6 小时注入一次外部 residual |
| 输出 | 解码到 37 个气压层上的温度、位势、u/v 风、比湿和两类云水 |
| 随机性 | 确定性 checkpoint 的 random field 为 `ZerosRandomField`，residual dropout 为 0 |
| 数据 | residual 训练 2015–2021，验证 2022，2023 预留测试 |
| forcing | 使用起报时刻前 24 小时的海温/海冰，整个 episode 持续使用该 forcing；时间特征随物理状态时间变化 |

NeuralGCM 是包含 learned physics 的完整混合模型，残差实验并未将其替换成裸物理求解器。浮点执行和跨 GPU 的精确复现仍需单独控制，确定性配置不等于所有硬件上天然逐 bit 相同。

### 3.2 Residual Graph–Mamba

残差分支继承 GraphCast 的图连接和 GNN 结构，使用 grid→mesh→mesh processor→grid，再接零初始化输出 head。这里的 GraphCast 只是空间网络结构，官方 GraphCast 预测器没有参与该 NeuralGCM baseline。

| 项目 | 设置 |
| --- | --- |
| 当前宽度 | 128 或 256 |
| Mesh | level 4 |
| Message-passing steps | 2 |
| Mamba | 每个 processor step 2 层，共 4 个 recurrent layers |
| Temporal dimensions | `d_inner=16`，`d_state=16`，`d_conv=4`，`dt_rank=32`，`bc_groups=1` |
| 初始化/精度 | Mamba1 initialization，fresh zero residual head，FP32，dropout 0 |
| 原生输入/输出通道 | `6 × 32 + 1 = 193`，包含 pressure 通道布局 |
| Known inputs | 地形/陆海、经纬度正余弦、滞后海温/海冰、年/日时间正余弦，共 12 个通道 |
| 归一化 | input features 与 residual increment 使用训练集统计；increment 乘各通道 correction scale，转换回求解器的无量纲球谐状态 |

当前 `no_pressure_zero_mean_v2` 的约束作用于增量：

- 禁止直接添加 log surface pressure residual，保留本步 NGCM 推进后的 pressure。
- 保留推进后状态原有的 degree-zero divergence/vorticity 系数，阻止分支额外注入这些模式。
- 仍允许修正其余温度、水分、非零阶散度/涡度模式。没有由这两项限制自动得到的完整能量、水分、正值性或动力平衡保证。

“本步不直接改 pressure”不等于“整条 residual 轨迹的 pressure 与原模型相同”。前一步的其他修正会改变下一次 NGCM 推进，因此后续 pressure 仍可能间接偏离。

### 3.3 一次 episode 的前向与梯度

```text
s0 = frozen_encoder(ERA5_at_origin, forcing)
h0 = 0

for k = 0 ... K-1:
    b_next = stop_gradient(NGCM_6h(stop_gradient(s_k), forcing))
    delta, h_next = residual(theta, h_k, stop_gradient(features(s_k)), known_k)
    s_next = constrained_add(b_next, correction_scale * delta)
    y_next = frozen_decoder(s_next, forcing)
    score y_next and s_next against the corresponding ERA5 targets
    feed stop_gradient(s_next) into the next physical step
```

中间状态没有被 ERA5 替换。未来 ERA5 只用来构造监督目标。分支读的是本步输入状态，而非先看到 NGCM 的下一步预测。

| 路径 | 是否反向传播 |
| --- | --- |
| 当前步 loss → decoder 输入 → residual 参数 | 是；decoder 参数仍冻结 |
| 后续 loss → Mamba memory recurrence → 较早分支参数 | 是，episode 内完整 memory BPTT |
| 后续 loss → 下一次物理积分 → 较早修正 | 否，physical feedback stop-gradient |
| 下一步分支 features → 前一步物理状态 | 否 |

这是用户指定的“冻结物理模型、只通过 memory 跨步求导”协议。它能降低反向开销，但其梯度并不是整个耦合物理预报系统的全导数。冻结 backbone 参数本身并不要求截断其输入梯度，这里两者都是显式选择。

一个需要特别注意的推论是：**当前 native log-pressure loss 对 residual 参数没有直接可用的梯度。** 本步 pressure 被恢复成 stopped baseline 的 pressure，前一步经动力学影响 pressure 的路径又被切断。因此该项可以记录越来越大的 pressure 误差，却不能沿这条物理因果路径指导残差修正。这是代码结构可确认的限制，尚不是新版 NaN 的单独因果证明。

旧版缓存 K=1 训练以 96 个连续真值编码记录为 segment、BPTT=24，跨记录携带 memory，但每条物理状态重新来自真值。新版每个 episode 从独立起点重置物理状态和 memory。新版 **K=1 没有跨两个预报时刻的 memory 训练链**，K=2 有一次跨步 memory 传递，K=20 有十九次，三者不能仅以同一个“K=1 预训练”标签理解。

## 4. 当前训练目标与训练配置

当前目标是对公开五项结构的显式重建：

```text
L = 20 M_data + M_model
    + 0.1 M_data_spectrum + 0.1 M_model_spectrum + 2 M_bias
```

- `M_data` 比较解码结果与 ERA5 的球谐系数。
- `M_model` 比较修正后的 native state 与冻结 encoder 编码的 ERA5 目标。
- 两项 spectrum loss 比较按 zonal wavenumber 求和后的谱幅度，保留 total wavenumber 0–42。
- Bias 项先在 batch 和 lead 上平均模态幅值差，再平方，含有跨样本/跨时间耦合。当前实现采用 `abs(modal)` 的幅值差。
- 字段求和，层数均匀平均，batch/time 平均，不计初始分析时刻。验证按固定 batch 计算目标，再按样本数加权平均，不能称为整套验证集统一计算的单个 bias 项。

loss scale 使用 2015–2021 中 **60 个训练时刻的 24 小时差分**。除比湿按层统计外，其余字段在空间、层和样本间池化。Native scale 由物理插值构造，native target 则由 learned encoder 构造，两者用途不同。Input/increment normalization 继续使用独立的原始训练统计，不能与 loss scale 混为一谈。

幅值因子在平方之前应用：位势 2、比湿 0.66、每类云水 0.05、native log pressure 5，其余 1。因此云水的平方因子是 **0.0025**。准确性/bias 的时间幅值为 `(1 + lead_hours/24)^(-1/2)`，谱项为 `(1 + (lead_hours/40)^4)^(-1/2)`。

当前滤波采用 `exp(-log(2) * (l/120)^24)`，在当前可表示波数范围内几乎是恒等变换。K20 仍使用这一短时近似，没有采用随 lead 改变的原论文拟合滤波。当前所有 37 层选择、均匀层平均、统计样本、稠密六小时间隔评分和滤波细节都有自己的定义，不能把结果称为原 NeuralGCM 训练方案的完整复现。公开依据见 [NeuralGCM 论文附录 7.3–7.5](https://arxiv.org/html/2311.07222v3#S7.SS3)。

| 参数 | 当前 K1/K2 | 当前 K20 |
| --- | --- | --- |
| 架构 | w128/w256，均 d16 | w128/d16 |
| Batch | 2 个独立起点 | 2 个独立起点 |
| Train/eval K | 每组匹配 1 或 2 | 均为 20 |
| Adam | β1=0.9，β2=0.95，ε=1e-6，无 weight decay | 相同，重新初始化 optimizer |
| Peak LR / warmup | 0.002 / 2,000 updates | 1e-5 / 100 updates |
| 配置预算 / review | 2,000 / 1,000 updates | 2,000 / 1,000 updates |
| 验证 | 每 100 updates，16 个固定 2022 起点 | 每 100 updates，16 个固定 2022 起点 |
| 初始化 | fresh zero head | K2/w128 最佳 update 800，仅转移 residual 参数 |
| 实际达到 | 各 1,000 updates | 100 updates 后失败 |

实际执行的 `TrajectoryTrainer` 使用直接 Adam，没有显式 global-gradient clipping。旧缓存 trainer 的 `clip_norm=1` 不适用于这批五项目标实验。数百万的梯度范数是风险信号，但 Adam 会按二阶矩归一化，不能把原始梯度范数直接当成参数步长，或单凭这一点认定学习率是唯一原因。

## 5. 旧版故障已经确认了什么

旧版三个失败作业完成第 5 轮后已保存 checkpoint，随后运行 20 步 cold validation。w128/di16、w128/di32、w256/di16 分别有 30/32、32/32、2/32 个起点失败。w256/di32 虽未出现 NaN，其五天误差也已极大。这里的 K20 是验证展开长度，不能误记为已经开始正式 K20 微调。

### 5.1 直接 pressure increment 导致第二步误差跃升

对旧 w128/di16、第 5 轮 checkpoint，在冬季起点仅于 +6 h 注入一次修正，之后不用第二次 residual，+12 h 的旧目标误差如下：

| +6 h 只施加一次的干预 | +12 h loss |
| --- | ---: |
| 不修正，冻结 NGCM | 17,204.66 |
| 全量 residual | 139,391.72 |
| 仅 pressure residual | 139,796.19 |
| 去掉 pressure，其余 residual 保留 | 19,404.57 |

这排除了“必须由第二次 Mamba memory 更新才会发生”的解释。该例实际 log-pressure 增量约为 -0.026 到 +0.075，相当于单次地面气压变化约 -2.57% 到 +7.82%。Pressure 同时影响求解器和 sigma→pressure 解码，短期解码分数获益并不保证状态能继续稳定积分。

### 5.2 额外零阶 divergence 是另一个独立机制

相同旧 checkpoint 的冬季 120 h 干预：

| 干预 | 结果 | 终点面积平均地面气压 |
| --- | --- | ---: |
| 原 NGCM | loss 21,197.03 | 98,560.64 Pa |
| 仅注入散度修正 | 114 h 非有限 | 无有效终点 |
| 同一散度修正，只去掉增量中的 l=0 | loss 23,215.79 | 98,560.77 Pa |
| 全 residual 去掉直接 pressure 与散度/涡度增量 l=0 | loss 38,808.25 | 98,560.45 Pa |

散度的垂直积分进入 pressure tendency。直接添加 arbitrary divergence 绕过了原 learned physics 从速度 tendency 转成 div/curl 的约束构造。限制必须针对新注入的系数，保留原 encoder/solver 已有系数。上述干预在两个旧 checkpoint、冬夏起点上消除了已观察到的灾难性漂移，但五天误差仍差于 baseline。

### 5.3 旧 loss 与进程退出是另外两件事

旧 loss 使用按层的 6 h 变化标准差，云水 scale floor 为 `1e-9`，完整验证审计中 baseline 总分超过 99.999% 来自云水。因此此前单步总分改善不能被解释成普遍气象技能改善。

旧 callback 对不合格 cold validation 调用 `select_checkpoint`，抛出的 `ValueError` 未被处理，训练因此退出。后续 legacy worker 已改为记录 failed origins、跳过不合格 selection，并继续其他验证。这个工程修复没有消除物理失稳本身，也不能解释新版 K20 在训练 batch 内部的异常。

八个历史诊断报告的 SHA256 在本次核查中全部匹配。完整的旧证据与范围见 [9/24 instability audit](https://github.com/Reminguch/Weather_Global/blob/3e736a632ae32a4bfea1e71f4fa8dc495f497adb/docs/experiments/neuralgcm_residual/INSTABILITY_AUDIT_20260924.md)。

## 6. 新版短时训练为什么“loss 很好，常用指标却差”

### 6.1 目标值确实下降

以下是同 horizon、同 16 个验证起点下的记录，不能跨 K 比绝对 loss 大小。

| K | Width | baseline loss | update 1000 loss | 最佳验证 loss | 最佳 update |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 5,808.064 | 142.437 | 132.107 | 900 |
| 1 | 256 | 5,808.064 | 148.071 | 129.157 | 900 |
| 2 | 128 | 5,341.049 | 183.791 | 157.452 | 800 |
| 2 | 256 | 5,341.049 | 188.379 | 151.650 | 900 |

### 6.2 同批验证的常用物理指标退化

以下使用各自最佳 checkpoint，报告 Gaussian-area gridpoint RMSE。Q700 换算为 g/kg，Z500 表示位势，单位为 m²/s²，没有除以重力加速度转换成高度。

| 指标 | 原 NGCM +6 h | K1/w128 +6 h | K1/w256 +6 h | K2/w128 +6 h | K2/w256 +6 h |
| --- | ---: | ---: | ---: | ---: | ---: |
| T850，K | 0.436 | 3.424 | 3.624 | 4.980 | 4.665 |
| Z500，m²/s² | 25.591 | 125.271 | 121.389 | 145.739 | 125.394 |
| U850，m/s | 0.862 | 2.200 | 2.247 | 3.425 | 2.545 |
| Q700，g/kg | 0.336 | 1.081 | 1.066 | 1.758 | 1.043 |

K2/w128 的 +12 h 指标也退化，例如 T850 为 **0.549 → 4.875 K**，Z500 为 **35.816 → 338.200 m²/s²**。另外一组 12 个共同起点的物理时间窗口也观察到同方向问题；那是连续起报的短预报窗口，不能当成一次连续三天 free-running 测试。

### 6.3 高层位势贡献解释了总分下降

以 K1/w128、第 900 步为例，位势共用的 24 h 差分 scale 是 **755.142 m²/s²**，所有 37 层等权，位势幅值再乘 2。原模型 +6 h 的 1 hPa 位势 RMSE 为 **30,017.113**，500 hPa 只有 **25.591**。

用保存的物理 RMSE 重建未滤波格点数据项，可得到数量级诊断：

```text
data_proxy(field) = 20 × mean_lead,level[
    amplitude(field)^2 × RMSE(lead, level)^2
    / scale(field, level)^2 / (1 + lead_hours/24)
]
```

| 未滤波格点数据项 proxy | 原 NGCM | K1/w128 最佳 |
| --- | ---: | ---: |
| 全部位势层 | 5,149.146 | 40.891 |
| 温度 | 9.484 | 20.445 |
| u 风 | 16.354 | 20.601 |
| v 风 | 2.114 | 9.964 |
| 比湿 | 80.774 | 32.431 |
| 全字段合计 | 5,257.888 | 124.588 |

原模型的位势 proxy 中，**99.70% 来自 1–10 hPa**。已有逐层分析还显示，1–30 hPa 位势约占原模型全部格点数据 proxy 的 **97.93%**。只要大幅消除这部分误差，其他气象指标即便恶化，总目标仍能显著下降。

这个 proxy 没有做球谐投影、滤波和 bias/spectrum 计算，**不是精确五项分解**。实际日志中的模态 data 项为 **5,254.410 → 120.878**，native model 项则从 **0.174 增到 2.477**。两套证据共同支持目标贡献失衡，不应把 proxy 的百分比标为精确完整 loss 占比。

在独立 12 起点窗口中，高层位势的巨大改善主要来自整层空间均值偏差的减少。例如 1 hPa 的 baseline MSE 中 98.65% 来自逐时空间均值偏差。该层并非完全没有空间细节改善，但总 RMSE 的大幅变化主要不能解释为非马尔可夫记忆带来的天气演变能力提升。

32 个均匀 sigma 层的最上层中心为 0.015625，地面气压约 1000 hPa 时相当于 15.6 hPa。1–10 hPa 通常落在层中心范围之外，decoder 涉及外推，因此高层表示和 decoder 值得重点核查。**这仍不能证明外推是唯一原因，更不能把另一分辨率模型的训练 mask 直接当成当前 2.8° 权重的训练 mask。** 20/30 hPa 的偏差也需要独立解释。[逐层偏差与物理时间证据](https://github.com/Reminguch/Weather_Global/blob/3e736a632ae32a4bfea1e71f4fa8dc495f497adb/plot/upper_air_physical_time_20260924/ANALYSIS.md)。

## 7. 当前 K20 失败的具体证据

### 7.1 转移正确，但起点本身不适合五天预报

K20 从 K2/w128/d16 的 best update 800 转移。该来源已完成 1,000 步 review，但其配置预算是 2,000 步。权重快照 SHA256 为：

```text
65932e949727d17129167ffeaea34d2466e74c854ae0b32c4894b9c41c5968d3
```

已有初始化审计确认生产 K20 的初始参数与该来源完全相同。Adam 与 schedule 重置，memory 每个起点重置。不能将失败归咎于“实际上没有加载到 K2 权重”。

| 固定 16 起点 K20 验证 | 五项 loss | 相对原模型 |
| --- | ---: | ---: |
| 原 NGCM | 2,477.110 | 1× |
| 刚载入 K2 权重，update 0 | 591,272.398 | 238.7× |
| 微调后 update 100 | 7,711,367.500 | 3,113.1× |

同一固定验证集上，100 步后的 loss 是转移起点的 **13.04 倍**。这些数字可以直接比较，因为 K、起点、loss 和统计相同。训练 batch 随 update 改变，其 loss 波动不能像这组固定验证结果一样解释。

| +120 h RMSE | 原 NGCM | 转移时 update 0 | K20 update 100 |
| --- | ---: | ---: | ---: |
| T850，K | 1.721 | 25.279 | 95.793 |
| Z500，m²/s² | 335.206 | 23,659.998 | 102,456.889 |
| U850，m/s | 3.357 | 74.593 | 309.382 |
| Q700，g/kg | 1.029 | 18.490 | 84.270 |

这些指标仍是有限值，但已经没有可用的预报精度。`finite=true` 不能当作稳定性或预报技能通过。

### 7.2 当前异常发生在训练，而非旧的 checkpoint selection

最后一条训练记录是 update 100，loss **8,232,452.5**，gradient norm **7,803,625.5**。其中 `data/geopotential` 贡献 **8,199,915.0，约占总训练 loss 的 99.60%**。新版这次的支配项已是位势，不应继续套用旧版的“云水几乎占全部 loss”结论。

随后完成 update 100 的固定验证和保存。`last.pkl` 内部 update=100，并保存了同一时刻的 optimizer/RNG。最后 traceback 是：

```text
trajectory_training.py:115
FloatingPointError: Nonfinite loss or gradient; no optimizer update applied
```

结合训练循环与保存的 `origin_order.json`，下一次待执行更新为 **101**，对应两个训练起点是 `2019-02-20T06`、`2015-05-03T18`。这两个起点是由确定的循环索引还原，并非异常日志已经写出了具体失败 origin。

现有日志没有区分以下情况：

- 前向 state/decoded output 已非有限，继而 loss 失败。
- 前向和 loss 有限，但某一步 decoder、谱项或 recurrent backward 产生非有限导数。
- 梯度叶子本身有限但某个诊断范数计算溢出。当前 fatal guard 检查的是 loss 与叶子，单独范数溢出并不足以触发这条异常。

因此下一次诊断必须保存每个 origin、lead、变量与反向阶段的有限性。当前 spectrum 实现使用 `safe_sqrt`，没有证据支持“普通 sqrt 在零点求导”已经是该次故障原因。

### 7.3 `best.pkl` 存在一个独立的选择漏洞

训练入口设置 `best=inf`，对转移起点会计算 `initial_validation.json`，但没有把初始 score 注册为 best。首次周期验证发生在 update 100，于是 **7,711,367.5** 被保存为 `best.pkl`，即使它显著差于 initial 的 **591,272.4**。

因此当前文件名 best 的含义只是“进入微调后，已记录周期验证中的最好结果”。它没有保护转移起点，也没有要求优于冻结 baseline。这个行为已由实际 checkpoint 内容和源码共同确认。它不是 NaN 的成因，但会误导后续模型选择。

## 8. 对当前原因的判断与证据强度

| 问题 | 证据强度 | 判断 |
| --- | --- | --- |
| 旧 pressure increment 与 l=0 divergence | 干预实验确认 | 解释旧版早期陡增和后期气压漂移；现版本已加入对应约束 |
| 全 37 层目标偏重高层位势 | 保存结果与逐项/逐层分析确认 | 当前总 loss 的改善不足以代表主要预报变量改善，checkpoint 排名标准需要修订 |
| K2 最佳权重直接用于 K20 时已很差 | 转移前固定验证确认 | 短时最佳 checkpoint 不能直接充当长时稳定起点 |
| Stop-gradient 截断物理因果链 | 冻结源码确认 | 训练不能沿动力学路径惩罚早期残差造成的后续损害；因果贡献大小需对照实验 |
| 当前 pressure native loss 的梯度盲区 | 约束与 SG 组合的结构性推论 | 指标能增长，物理误差路径的梯度却被截断，需要专门确认与改进训练信号 |
| K2→K20 的 memory/state 分布变化 | 明确的实验差异，因果性未确认 | memory 从两步延长到二十步，状态也偏离短时分布，值得做 memory reset 与固定状态对照 |
| 其他非零模式、温湿场、负云水引发失稳 | 有可疑物理指标，未定位到本次首个非有限值 | 现有约束不构成完整守恒或正值性，必须逐变量记录，不能猜测唯一变量 |
| LR/无 clipping/gradient amplification | 风险信号，未有独立消融 | 可测量更新量、分层梯度与固定 batch 改善，不应直接声称降 LR 或 clipping 必然解决 |
| GPU、随机 cache 或 requeue 损坏 | 无当前直接支持 | 确定性 backbone、精确转移及 resume 测试已有证据；仍需对失败 batch 定点复现才能排除所有数值实现因素 |
| 初始权重未纳入 best selection | 源码与 checkpoint 确认 | 应同时保留 initial 与 trained best；有限但极差的模型不能被误称为改进 |

独立 smoke 确认了加载、真实更新、有限输出、恢复和 checkpoint 生命周期，且用了隔离 pilot。它们没有证明经过数百次优化后的生产权重能在所有季节持续 20 步稳定。先前把 smoke 通过和总 loss 下降作为继续长时训练的充分依据，缺少了真实来源 checkpoint 的长时技能与逐变量验收。

## 9. 建议的下一步，按优先级排列

### P0：先恢复可诊断性和可信模型选择

1. 固定当前 update-100 checkpoint、两个待执行起点、RNG 和统计，运行独立的失败 batch 复现。分别记录 corrected state、decode、每项 loss、loss cotangent、每层 backward gradient 的首个非有限位置。保留 temperature、pressure、水分范围和 residual/native scale 比值。
2. 同时运行原 NGCM、初始 K2 权重和 update-100 权重的固定起点对照。先区分权重导致的问题与某个样本/实现问题，再决定修正。
3. 将 initial checkpoint 纳入选择候选，分别记录“最好目标分数”和“满足关键物理指标要求的最好模型”。诊断失败要输出上下文，不能通过跳过坏 origin 来制造漂亮平均值。

### P1：修正训练目标与有效层范围的错配

1. 对当前官方 2.8° checkpoint 做 encode/decode 与逐层 baseline 审计，核查上层外推、learned decoder、单位、数据预处理和原始有效层范围。保留 20/30 hPa 的单独分析。
2. 预先确定主指标和物理有效层范围。保留全部层报告，同时用独立对照检验层权重/选择、变量尺度和网格空间损失。不能只在旧权重上重算分数就声称重训已有效。
3. 选择 checkpoint 时报告 T850、Z500、风场、湿度、云水与高层位势，防止某一项巨大改进抵消关键指标的系统性退化。

### P2：验证长时反馈与梯度方案

1. 先对真实 K2 生产 checkpoint 做长时 probe，再检验逐步增长 K 的课程，例如 2→4→8→20。当前 `PaperLoss` 仅接受 1/2/20，实施中间 K 需要显式扩展并测试，不能只改命令行。
2. 比较固定权重的单次 residual impulse、逐步 residual、memory reset、分通道屏蔽和残差幅度缩放，定位状态反馈与 memory 的相对作用。
3. 如继续采用 physical SG，考虑增加可直接约束 native balance、增量幅度及稳定性的目标，并明确它仍是 surrogate gradient。另做小规模保留 solver 输入导数的短链对照，可检验缺失物理梯度的重要性，但它属于协议变更。
4. 对 optimizer 记录相对参数更新量和各模块梯度，再单独检验 LR、warmup、clipping 或小幅 residual gate。每项变更独立编号，避免同时改多项后无法判断作用。

新 GPU 诊断须使用独立 Slurm `gpu-test` 作业，默认不超过一小时，由提交策略选择 partition，不显式指定 partition。测试产物与生产 checkpoint 分开。训练结果图的横轴使用 optimizer update count。

本次 9/29 工作完成的是只读核查、重新汇总与报告发布，没有执行上述新 GPU 对照，也没有修复或重启训练。建议项不是通过的实验结果。

## 10. 证据定位与复核

### 10.1 文件位置

运行产物位于计算节点代码工作区 `/home/sh4809/weatherforecast/Weather_Global`。下表路径相对于该目录。这些运行日志/冻结快照不在本文所在的论文仓库中，因此以原始路径和 hash 定位，而非提供失效的相对链接。

| 证据 | 相对路径 |
| --- | --- |
| 旧干预汇总 | `docs/experiments/neuralgcm_residual/instability_audit_20260924.json` |
| 八个旧干预原报告 | `logs/neuralgcm_instability_20260924/`，本次 8/8 hash 匹配 |
| K1/K2 实际数值源码 | `logs/ngcm_aligned_20260924/source_transfer_v2/` |
| K20 实际执行源码 | `logs/ngcm_aligned_20260924/source_k2_transfer_v1/` |
| 五项目标统计 | `logs/ngcm_aligned_20260924/shared_statistics_cpu_v1/statistics.json` |
| 四组训练 | `logs/ngcm_aligned_20260924/train_k{1,2}_w{128,256}_d16_slices_v1/` |
| K20 当前产物 | `logs/ngcm_aligned_20260924/finetune_k20_from_k2_w128_d16_slices_v1/` |
| K20 stderr | `logs/ngcm_aligned_20260924/ngcm-k20-from-k2-w128-d16-14387208.err` |
| 小窗口物理量评估 | `plot/paper_physical_time_20260924/` |
| 高层偏差分解 | `plot/upper_air_physical_time_20260924/` |

K1/K2 与 K20 快照中的 `corrections.py`、`trajectory_training.py`、`paper_loss.py`、`paper_statistics.py`、`model.py`、`backbone.py` 和 `runtime.py` 本次比较内容相同。工作树存在其他未提交改动，因此本报告以冻结执行快照为准，不能仅用工作区 Git HEAD 代表所有生产实现。

关键源码位置以 `source_k2_transfer_v1/` 为根：

| 文件与行号 | 作用 |
| --- | --- |
| `src/models/neuralgcm_residual/corrections.py:17–26` | pressure 与零阶增量约束 |
| `src/models/neuralgcm_residual/trajectory_training.py:23–28` | 实际 Adam 和 schedule |
| `src/models/neuralgcm_residual/trajectory_training.py:68–90` | 前向闭环、SG 和目标 |
| `src/models/neuralgcm_residual/trajectory_training.py:104–115` | memory reverse pass 与 fatal finite guard |
| `src/models/neuralgcm_residual/paper_loss.py:77–106` | 五项目标与字段分解 |
| `scripts/training/train_neuralgcm_paper_residual.py:189–191,236–247,265–273` | best 初始化、初始验证、周期 selection |

### 10.2 关键 SHA256

以下运行产物路径相对于 `logs/ngcm_aligned_20260924/`：

| 文件 | SHA256 |
| --- | --- |
| `shared_statistics_cpu_v1/statistics.json` | `55f59a53463facad8a59ee0c3e91217d2c8523a8dd8bcb3ee146c2e56b86a82e` |
| `train_k1_w128_d16_slices_v1/baseline_validation.json` | `8c0208338c93ec6d71762a0b9167b539252b3742f718c79370b43e58dd0dcd61` |
| `train_k1_w128_d16_slices_v1/validation.jsonl` | `dddd3dbc0cc54cf312b659825e2e890a92681209a0a6d250d091602a4adef59d` |
| `train_k2_w128_d16_slices_v1/validation.jsonl` | `6daeed6f71f1ff44d9103e66bce13d4e0352839dd0778014a09bf594ead65500` |
| `finetune_k20_from_k2_w128_d16_slices_v1/initial_validation.json` | `6b6009ba3018a6428be633bab30b728cb4fbe990d52ab4869ef2adea069e7245` |
| `finetune_k20_from_k2_w128_d16_slices_v1/validation.jsonl` | `007517b40b85c7aef3688e9116df66d0fa91b02677f939e6affed062eaa0b2d7` |
| `finetune_k20_from_k2_w128_d16_slices_v1/metrics.jsonl` | `44b0910ea99ee7197477bcb085283f47186cfb6051fbd88a2edf685e7086af40` |
| `finetune_k20_from_k2_w128_d16_slices_v1/last.pkl` | `8daa804f8178921a8ede8e89a2c9e20ba47f853711f8f85aa4ca7cadc6319334` |
| `source_k2_transfer_v1/src/models/neuralgcm_residual/trajectory_training.py` | `3783074fc6c888757720b2303ddd1001bc8edb46f048ff9632e5208cd304f218` |
| `source_k2_transfer_v1/src/models/neuralgcm_residual/paper_loss.py` | `83a71dad358d3e3ac1b2ac673e3d385531b82ec0a95177a14e9b8b7990bca367` |
| `source_k2_transfer_v1/src/models/neuralgcm_residual/corrections.py` | `757f85b7451bfddc087bb3df02d2d6c5aae9250834d50720b66ff0f28433f5eb` |

### 10.3 最小只读复核

在代码工作区运行以下 CPU 检查，可核对 K20 的初始/最后 loss、最后成功 update 和下一批起点，不会运行模型或修改训练产物：

```python
import json
from pathlib import Path

root = Path('logs/ngcm_aligned_20260924')
run = root / 'finetune_k20_from_k2_w128_d16_slices_v1'
baseline = json.loads((run / 'baseline_validation.json').read_text())
initial = json.loads((run / 'initial_validation.json').read_text())
validation = [json.loads(s) for s in (run / 'validation.jsonl').read_text().splitlines()]
metrics = [json.loads(s) for s in (run / 'metrics.jsonl').read_text().splitlines()]
origins = json.loads((run / 'origin_order.json').read_text())
completed = metrics[-1]['update']
print('baseline, initial, last validation:',
      baseline['loss'], initial['loss'], validation[-1]['loss'])
print('completed updates:', completed)
print('next batch:', [origins[(completed * 2 + i) % len(origins)] for i in range(2)])
```

预期输出依次为 `2477.1099548339844`、`591272.3984375`、`7711367.5`，最后成功 update 为 `100`，下一批为 `2019-02-20T06` 和 `2015-05-03T18`。其中“第 101 次更新为什么首先非有限”仍需要前述定点 GPU 诊断来回答。
