#!/usr/bin/env python3
"""Decompose saved geopotential MSE into instantaneous spatial bias and remainder."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parent
    output = root / 'upper_air_physical_time_20260924'
    receipt = json.loads((output / 'provenance.json').read_text())
    source = output / 'physical_errors.csv'
    if hashlib.sha256(source.read_bytes()).hexdigest() != receipt['files'][source.name]:
        raise ValueError('Input checksum mismatch')
    with source.open() as stream:
        errors = list(csv.DictReader(stream))
    groups = {}
    for row in errors:
        if row['field'] != 'geopotential':
            continue
        key = (row['run_id'], int(row['lead_hours']), int(float(row['pressure_hpa'])))
        groups.setdefault(key, []).append(row)
    rows = []
    for (run, lead, pressure), samples in sorted(groups.items()):
        item = dict(run_id=run, lead_hours=lead, pressure_hpa=pressure)
        for model in ('baseline', 'residual'):
            mse = np.asarray([float(r[model + '_mse']) for r in samples])
            bias = np.asarray([float(r[model + '_bias']) for r in samples])
            centered = mse - bias ** 2
            if np.any(centered < -1e-8 * np.maximum(mse, 1)):
                raise ValueError('Spatial bias energy exceeds MSE')
            item[model + '_mse'] = float(mse.mean())
            item[model + '_rmse'] = float(np.sqrt(mse.mean()))
            item[model + '_mean_bias'] = float(bias.mean())
            item[model + '_spatial_bias_energy'] = float((bias ** 2).mean())
            item[model + '_spatial_bias_energy_pct'] = float(100 * (bias ** 2).mean() / mse.mean())
            item[model + '_centered_rmse'] = float(np.sqrt(max(0., centered.mean())))
        gain = item['baseline_mse'] - item['residual_mse']
        bias_gain = item['baseline_spatial_bias_energy'] - item['residual_spatial_bias_energy']
        item['mse_reduction'] = gain
        item['bias_removal_share_of_mse_reduction_pct'] = 100 * bias_gain / gain if gain != 0 else None
        rows.append(item)
    path = output / 'geopotential_bias_decomposition.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = json.loads((output / 'loss_decay_audit.json').read_text())
    w128 = {r['pressure_hpa']: r for r in rows if r['run_id'] == 'k1_w128_d16' and r['lead_hours'] == 6}
    z1 = w128[1]
    lines = ['# 为什么高层位势改善，但常用物理指标变差？', '',
        '**结论：高层位势的改善是真实测量结果。当前 loss 大幅下降主要来自修正高层的巨大系统偏差，不能解释为所有气压层的预报都改善。**', '',
        '[物理量随时间变化及逐时 RMSE 总图](geopotential_overview.png) · [全部图表与数据](README.md)', '',
        '## 1. 改善在哪些层？', '',
        '在同一批 12 个起报时刻上，K1/K2、w128/w256 的所有组合，都在 1、2、3、5、7、10、20、30 hPa 的位势上改善。',
        '以下展示 K1/w128、+6 小时，选用验证 loss 最优的第 900 步 checkpoint。所有数值单位均为 m²/s²。', '',
        '| 气压层 | baseline RMSE | residual RMSE | baseline 空间均值偏差 | residual 空间均值偏差 | baseline MSE 中空间均值偏差的占比 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for level in (1, 2, 3, 5, 7, 10, 20, 30, 500):
        r = w128[level]
        lines.append(f"| {level} hPa | {r['baseline_rmse']:,.1f} | {r['residual_rmse']:,.1f} | "
                     f"{r['baseline_mean_bias']:,.1f} | {r['residual_mean_bias']:,.1f} | {r['baseline_spatial_bias_energy_pct']:.2f}% |")
    lines += ['', '这里的 RMSE 是先计算每个格点误差，再作高斯面积和起报时刻平均，最后开平方。它不是全球均值曲线之间的 RMSE。',
        '“没有任何变量改善”不符合完整数据。之前七个常用变量/层组合都恶化，不能外推为所有 37 层都恶化。', '',
        '## 2. 高层改进主要是在消除整体偏差', '',
        '对每个预报时刻，令 e 是格点误差、b 是它的面积平均，有精确分解：', '',
        '```text', 'area_mean(e²) = b² + area_mean((e - b)²)', '```', '',
        '再对 12 个时刻平均，得到 CSV 中的空间均值偏差能量和去掉该偏差后的剩余误差。',
        '这不同于先对整个窗口平均偏差再平方，也不是训练中的模态 bias loss。', '',
        f"在 1 hPa，baseline 空间均值偏差约 **{z1['baseline_mean_bias']:,.1f}**，residual 约 **{z1['residual_mean_bias']:,.1f}**。",
        f"baseline 的 **{z1['baseline_spatial_bias_energy_pct']:.2f}%** MSE 来自每个时刻的空间均值偏差。",
        f"该层 MSE 减少的 **{z1['bias_removal_share_of_mse_reduction_pct']:.2f}%** 可以由空间均值偏差能量的减少解释。",
        f"去掉每个时刻的空间均值偏差后，RMSE 为 **{z1['baseline_centered_rmse']:,.1f} → {z1['residual_centered_rmse']:,.1f}**。",
        '因此总 RMSE 的巨大改善主要是消除了整层偏移，并不等于天气随时间变化的细节同幅度改善。', '',
        '## 3. 为什么这部分主导 loss decay？', '',
        '为了和已记录的 loss 严格对应，这一节使用原训练的 16 个固定验证起报时刻，与上面画图用的 12 个起报时刻分开。',
        '同一个 K1/w128、第 900 步 checkpoint，真实记录的五项总 loss 为：', '',
        f"**{audit['exact_loss']['baseline']:.3f} → {audit['exact_loss']['residual']:.3f}**。", '',
        '当前实现对全部 37 层等权平均，位势共用一个 24 小时差分标准差约 755.14 m²/s²，位势幅值再乘 2。',
        '因此位势的平方误差会额外乘 4。标准差反映训练数据的变化尺度，不保证 baseline 的各变量误差贡献相等。',
        '对于一个 +6 小时预报，忽略球谐投影和滤波后的数据项近似为：', '',
        '```text', '20 / 37 × Σ_level [0.8 × (2 × RMSE_level / 755.14)²]', '```', '',
        '相同验证数据的未滤波格点诊断显示，1–30 hPa 位势占 baseline 数据项约 **97.93%**。',
        f"其贡献从 **{audit['upper_geopotential']['baseline']:.3f}** 降到 **{audit['upper_geopotential']['residual']:.3f}**。",
        '相比之下，温度项从 9.48 增至 20.45、南北风从 2.11 增至 9.96，远不足以抵消这个下降。',
        '比湿项也从 80.77 降至 32.43，改善集中在少数高层。', '',
        '**这些格点诊断只用于解释数量级，不是精确的五项 loss 分解。** 精确模态 data 项为 5254.410 → 120.878。',
        '同一日志的 native model 项反而从 0.174 增至 2.477，说明总分改善并不意味着所有子项都改善。', '',
        '## 4. baseline 的高层偏差为什么大？哪些原因仍是假设？', '',
        '已确认的实现事实：', '',
        '- 当前官方 2.8° 推理配置使用 32 个等距 sigma 层。代码定义的最上层中心为 sigma=0.015625。若地面气压约为 1000 hPa，它对应约 15.6 hPa。',
        '- decoder 先从温度、湿度诊断位势，再从 sigma 坐标映射到气压坐标。公开代码对位势和温度采用线性插值及范围外线性外推，然后施加 learned decoder correction。',
        '- 因而 1–10 hPa 等输出通常位于原生层中心范围之外。输出接口包含这些气压层，并不意味着它们拥有同样充分的原生垂直分辨率。20、30 hPa 的误差不能仅用范围外外推解释。',
        '- 官方补充材料 §I.1（PDF 第 55 页，正文页码 54）明确讨论了 **1.4°** 模型在 30 hPa 以上的偏差，并说明这些层未被优化。这是核查高层有效范围的重要线索，但不能直接当作当前 **2.8°** checkpoint 原始训练 mask 的证明。', '',
        '参考：[官方补充材料 §B.1、§D.2、§I.1](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-07744-y/MediaObjects/41586_2024_7744_MOESM1_ESM.pdf)、',
        '[decoder 实现](../../third_party/neuralgcm/neuralgcm/legacy/decoders.py)、',
        '[2.8° 配置](../../third_party/neuralgcm/neuralgcm/reference_code/paper_configs/deterministic_2_8_deg.gin)。', '',
        '**基于这些事实的推断**：有限的高层垂直表示、外推和 decoder 行为，很可能是高层系统偏差的重要来源。',
        '目前没有运行独立的 encode/decode 分解、关闭 learned decoder correction 或真实模式层诊断，不能把任一项确定为唯一原因，也没有排除数据处理问题。', '',
        '## 5. 为什么修正高层会伴随其他层恶化？', '',
        '残差分支修正的是共享原生大气状态；当前 loss 允许高层大幅得分覆盖其他变量或气压层的退化。',
        '现有记录直接证明了这种取舍。共享状态和冻结 decoder 会使不同输出彼此耦合，',
        '但“具体哪条梯度路径导致其他层恶化”仍需梯度或消融实验，不能单凭曲线断言。', '',
        '当时选用五项 loss 是为了借鉴论文对准确性、谱和偏差的联合约束。',
        '但当前的全部 37 层选择、层平均、统计量和滤波是重建实现的一部分；公式外形一致并不代表完整复现了论文训练配置。',
        '在没有核实有效层范围和逐项贡献前，将它用于残差训练并把总 loss 下降解读成普遍技能提升，是此前分析的问题。', '',
        '下一步有针对性的验证是：对同一批起报时刻分解官方模型的 encode/decode 和逐层误差，核实 2.8° 原始有效层范围，',
        '再用保持数据和优化器设置不变的独立对照实验检验层选择或权重。仅在旧 checkpoint 上重新打分不能证明重训会改善。',
        '本次提交只补充图表与分析，没有修改训练目标或重新启动训练。', '',
        '[完整偏差分解 CSV](geopotential_bias_decomposition.csv) · [loss 诊断 JSON](loss_decay_audit.json)', '']
    (output / 'ANALYSIS.md').write_text('\n'.join(lines))
    note = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        method='Per-origin spatial MSE = spatial-mean-error squared + centered spatial MSE; then time mean',
        rows=len(rows), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files={name:hashlib.sha256((output / name).read_bytes()).hexdigest() for name in
               ('geopotential_bias_decomposition.csv', 'ANALYSIS.md')})
    (output / 'analysis_provenance.json').write_text(json.dumps(note, indent=2) + '\n')
    print(json.dumps(z1, indent=2))


if __name__ == '__main__':
    main()
