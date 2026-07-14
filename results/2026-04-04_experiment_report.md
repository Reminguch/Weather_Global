# Mamba-GraphCast 实验报告 (2026-04-04)

## 一、我们改了什么

### 1.1 在 GraphCast 里加了一个 Mamba 时序模块

GraphCast 原本的预测方式是：拿最近 2 帧（t-1, t）→ 预测下一帧（t+1）→ 把预测喂回去继续预测。每一步都只看最近 2 帧，没有任何"记忆"。

我们在 GraphCast 的 mesh processor 之前插入了一个 Mamba SSM 模块，让模型在做多步预报时能"记住"之前步骤的信息。

### 1.2 三个版本的演进

**Version 1: Stateless Mamba** (`temporal_mesh_mamba.py`)
- 改了编码路径：从"拼接所有帧的 channel"改成"每帧单独编码"
- Mamba 处理 2-8 帧的时间序列
- 每次前向传播 hidden state 清零，没有跨步记忆

**Version 2: Stateful Mamba** (`temporal_mesh_mamba_stateful.py`)
- 同样改了编码路径
- 通过 `hk.get_state`/`hk.set_state` 让 hidden state 跨 rollout step 传递
- 每个训练样本开始时清零

**Version 3: Residual Memory** (最新，复用 `temporal_mesh_mamba_stateful.py`)
- **不改编码路径**，跟 baseline 100% 相同
- Mamba 只接收单帧 mesh latent `[mesh, batch, D]`，通过 hidden state 注入跨步记忆
- 输出通过残差连接加回去：`output = input + proj(mamba_output)`

---

## 二、发现了什么问题

### 问题 1: Transpose Bug（Version 1）

`temporal_mesh_mamba.py` 中 `_run_sequence` 的输出有一个错误的 transpose：

```python
# 错误：多了一次 transpose，把 mesh 和 batch 维度搞反了
return jnp.transpose(sequence[-1], (1, 0, 2))

# 正确：sequence[-1] 本身已经是 [n_mesh, batch, channels]
return sequence[-1]
```

**后果**: 所有 Phase 1 的 stateless Mamba 实验输出的 mesh node 和 batch 维度是反的。下游 GNN 足够鲁棒所以 loss 还能降，但 Mamba 的输出实际上是乱的。这解释了为什么 stateless Mamba 跟 baseline 几乎一样——它的输出没有提供有意义的信息，GNN 在"忽略"它。

### 问题 2: 编码路径不公平（Version 1 & 2）

当 `temporal_backbone=mamba` 时，模型走一条完全不同的编码路径：

```
Baseline:                              Mamba (旧):
  输入 [t-1, t]                          输入 [t-1, t]
    ↓                                      ↓
  拼接 channel → [grid, batch, C*2]      逐帧编码 → [grid, batch, C] × 2
    ↓                                      ↓
  Grid2Mesh (1 次) → [mesh, batch, D]    Grid2Mesh (2 次) → [T, mesh, batch, D]
    ↓                                      ↓
  直接进 Mesh GNN                         Mamba 处理时间轴 → [mesh, batch, D]
                                           ↓
                                         进 Mesh GNN
```

**后果**: 实验中 baseline 和 Mamba 有两个变量不同：(1) 编码路径 (2) Mamba 模块。无法判断性能差异来自哪个。

而且 baseline 的 channel 拼接让 Grid2Mesh GNN 能在一次 MLP 中联合处理两帧，直接学到速度/加速度等时序特征。逐帧编码强迫每帧独立处理，丢失了这个优势。**编码路径本身可能就是更差的选择**。

### 问题 3: Mamba 做了重复的工作（Version 1 & 2）

Baseline 通过 channel 拼接已经编码了 2 帧的时序信息。Mamba 放在编码后面，又在这 2 帧上做一遍时序处理——本质上跟 channel 拼接做同样的事。多了参数和优化负担，却没有新信息。

### 问题 4: Rollout 太短（Version 2）

Stateful Mamba 的核心价值是跨 rollout step 的长程记忆。但训练时 `target_steps` 只有 2 或 4，意味着 Mamba hidden state 最多传递 2-4 步。这么短的序列里，2 帧输入已经提供了足够的时序信息，hidden state 的边际价值极低。

---

## 三、实验结果

### Phase 1: Stateless Mamba (mesh_size=3, batch=4, target=1, 20k steps)

| Model | RMSE @8k | MAE @8k | vs Baseline |
|-------|----------|---------|-------------|
| **Baseline (input=2)** | **138.55** | **35.13** | — |
| Mamba h2 (input=2) | 139.39 | 35.47 | +0.6% |
| Mamba h4 (input=4) | 138.83 | 35.39 | +0.2% |

**结论**: 无提升。受 transpose bug + 冗余处理影响。

### Phase 2: Baseline 对照 (mesh_size=4, batch=1, 20k steps)

| input_steps | target_steps | RMSE @20k | MAE @20k |
|-------------|-------------|-----------|----------|
| 2 | 2 | 62.41 | 18.21 |
| **4** | **2** | **59.59** | **17.48** |
| 2 | 4 | 96.08 | 26.89 |
| 4 | 4 | 95.13 | 26.71 |

**发现**:
- input_steps=4 比 2 好 ~5%（更长历史有帮助）
- target_steps=2 远优于 4（长 rollout 训练更难，loss 更大）

### Phase 2: Stateful Mamba — 旧编码路径 (mesh_size=4, batch=1, 已取消)

| Config | Best RMSE | Eval Step | Baseline @20k | 差距 |
|--------|-----------|-----------|---------------|------|
| sf-h2-t2 | 72.70 | @14k | 62.41 | +16.5% |
| sf-h4-t2 | 71.37 | @12k | 59.59 | +19.8% |
| sf-h2-t4 | 108.41 | @12k | 96.08 | +12.8% |
| sf-h4-t4 | 111.09 | @10k | 95.13 | +16.8% |

**结论**: 全面落后 baseline 13-20%。在约 70% 进度时取消，因为追上无望且编码路径不公平导致结果不可解释。

---

## 四、新方案: Residual Memory

### 4.1 设计原则

针对上述所有问题的修复：

| 问题 | 旧方案 | 新方案 |
|------|--------|--------|
| 编码路径不公平 | 逐帧编码 | **跟 baseline 完全一样**（channel 拼接） |
| Mamba 做重复工作 | 处理 2-8 帧序列 | **只处理 1 帧 + hidden state** |
| Rollout 太短 | target=2/4 | **测试 target=2/4/6** |
| 无法归因 | 2 个变量混淆 | **唯一区别就是 Mamba 残差块** |

### 4.2 架构

```
Baseline:                              Residual Memory:

  Channel 拼接 [t-1, t]                  Channel 拼接 [t-1, t]        ← 相同
    ↓                                      ↓
  Grid2Mesh (1次)                         Grid2Mesh (1次)              ← 相同
    ↓                                      ↓
  mesh_latent [mesh, batch, D]            mesh_latent [mesh, batch, D]
    ↓                                      ↓
    │                                    ★ Mamba: load h_prev          ← 唯一区别
    │                                    │  h_new = decay*h + (1-decay)*u
    │                                    │  save h_new
    │                                    │  output = input + residual
    ↓                                      ↓
  Mesh GNN                               Mesh GNN                     ← 相同
    ↓                                      ↓
  Mesh2Grid                               Mesh2Grid                    ← 相同
    ↓                                      ↓
  预测 t+1                                预测 t+1
```

### 4.3 跨 Rollout Step 的状态传递

```
训练样本开始 → h = 0

Step 1: input=[t-1, t] → 编码 → mesh_latent
        Mamba: h=0 → h₁, output = mesh_latent + residual₁
        → 预测 t+1

Step 2: input=[t, t+1] → 编码 → mesh_latent
        Mamba: h₁ → h₂, output = mesh_latent + residual₂
        （h₂ 包含 step 1 的信息）
        → 预测 t+2

  ...以此类推...

训练样本结束 → 下个样本 h = 0
```

### 4.4 代码改动

**`graphcast.py`** — `__call__` 方法:
```python
# 新增 mesh_post_encoder_residual: 用 baseline 编码 + Mamba 残差注入
if self._temporal_backbone == "none" or self._temporal_location == "mesh_post_encoder_residual":
    grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)  # 跟 baseline 一样
    (latent_mesh_nodes, latent_grid_nodes) = self._run_grid2mesh_gnn(grid_node_features)
    if self._temporal_location == "mesh_post_encoder_residual":
        latent_mesh_nodes = self._run_temporal_mesh_block(latent_mesh_nodes, is_training=is_training)
```

**`temporal_mesh_mamba_stateful.py`** — `TemporalMeshBlock.__call__`:
```python
if mesh_latent_tnbd.ndim == 3:
    # [n_mesh, batch, D] → 加 time=1 维度 → Mamba 处理 1 步 + state → 去掉 time 维度
    mesh_4d = mesh_latent_tnbd[None]       # [1, mesh, batch, D]
    out_4d = self._run_sequence(mesh_4d, is_training=is_training)
    return out_4d[0]                        # [mesh, batch, D]
```

### 4.5 实验设计

6 组对照实验（500 步 smoke test，job 6527518，排队中）：

| | target=2 | target=4 | target=6 |
|---|---|---|---|
| **Baseline** | baseline_t2 | baseline_t4 | baseline_t6 |
| **Residual Memory** | resmem_t2 | resmem_t4 | resmem_t6 |

**如何判读结果**:
- resmem_tX vs baseline_tX → Mamba 在该 rollout 长度下是否有帮助
- resmem 的优势是否随 target_steps 增大而增大 → 长程记忆是否有价值
- resmem 是否至少不比 baseline 差 → 残差连接是否保底

---

## 五、时间线

| 日期 | 事件 |
|------|------|
| 04-01 | Phase 1 stateless Mamba 实验完成，发现无提升 |
| 04-02 | 发现 transpose bug，开发 stateful Mamba |
| 04-03 | 提交 Phase 2 baseline + stateful 对照实验 |
| 04-03 | 早期 stateful 结果（旧批次）显示 RMSE -9.7%，但不公平对比 |
| 04-04 | Baseline 20k 完成，stateful 全面落后 baseline 13-20% |
| 04-04 | 诊断出编码路径差异是主要问题 |
| 04-04 | 设计 residual memory 方案，修复所有已知问题 |
| 04-04 | 提交 target_steps=10 长 rollout 实验（旧编码路径） |
| 04-04 | 取消旧 stateful 实验，提交 residual memory smoke test |
