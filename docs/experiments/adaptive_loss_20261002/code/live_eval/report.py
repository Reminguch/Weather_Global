"""Render every physical output variable against cumulative training compute time."""
import argparse,csv,fcntl,io,json,os,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import FIELDS,LABELS,read,write,atomic,reduce_fields

COLORS=dict(fixed='#2467ae',calibrated='#e78a28',adaptive='#1b9470')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;plan=read(root/'plan.json')
    lock=(root/'.report.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    records=[];failures=[]
    for path in sorted((root/'history').glob('*.json')):
        history=read(path);baseline=history['validations'][0]['physical_rmse']
        for row in history['validations']:
            for lead_index,lead in enumerate(row['lead_hours']):
                for band in ('all_levels','weather_gt30hpa'):
                    summary=reduce_fields(row['physical_rmse'],baseline,row['pressure_hpa'],lead_index,band,plan['threshold_percent'])
                    records.append(dict(scope='short',hardware=history['hardware'],mode=history['mode'],update=row['update'],training_compute_seconds=history['cumulative_step_seconds'][row['update']],lead_hours=lead,**summary))
    for path in sorted((root/'results').glob('*.json')):
        row=read(path)
        if not row['finite']:
            failures.append({k:row[k] for k in ('id','hardware','mode','update','error_type','message')});continue
        baseline=read(root/'baselines'/Path(row['baseline_path']).name)['physical_rmse']
        for lead in (6,12,24,72,120):
            index=row['lead_hours'].index(lead)
            for band in ('all_levels','weather_gt30hpa'):
                summary=reduce_fields(row['physical_rmse'],baseline,row['pressure_hpa'],index,band,plan['threshold_percent'])
                records.append(dict(scope='long',hardware=row['hardware'],mode=row['mode'],update=row['update'],training_compute_seconds=row['training_compute_seconds'],lead_hours=lead,snapshot_id=row['id'],**summary))
    write(root/'improvement_history.json',dict(records=records,failed_checkpoints=failures,generated_at=time.time(),time_axis=plan['time_axis']))
    stream=io.StringIO();keys=['scope','hardware','mode','update','training_compute_seconds','lead_hours','band','variable','rmse','baseline_rmse','improvement_percent']
    writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
    for r in records:
        for field,value in r['fields'].items():
            row={k:r[k] for k in keys if k in r};row.update(variable=field,**{k:value[k] for k in ('rmse','baseline_rmse','improvement_percent')});writer.writerow(row)
    atomic(root/'variable_improvement.csv',stream.getvalue().encode())
    hardware=sorted({r['hardware'] for r in records});plots=[]
    for hw in hardware:
        for scope,lead in [('short',12),('long',120)]:
            for band in ('all_levels','weather_gt30hpa'):
                selected=[r for r in records if r['hardware']==hw and r['scope']==scope and r['lead_hours']==lead and r['band']==band]
                if not selected:continue
                fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
                for field,label,ax in zip(FIELDS,LABELS,axes.flat):
                    for mode,color in COLORS.items():
                        rows=sorted((r for r in selected if r['mode']==mode),key=lambda r:r['update'])
                        xs=[r['training_compute_seconds']/3600 for r in rows];ys=[r['fields'][field]['improvement_percent'] for r in rows]
                        if scope=='long':xs=[0.]+xs;ys=[0.]+ys
                        if rows:ax.plot(xs,[np.nan if y is None else y for y in ys],color=color,label=mode,marker='.',lw=1.6)
                    ax.axhline(0,color='black',ls='--',lw=.8);ax.set(title=label,xlabel='Training compute time (hours)',ylabel='RMSE improvement (%)');ax.grid(alpha=.2)
                last=axes.flat[-1];last.axis('off')
                handles,labels=axes.flat[0].get_legend_handles_labels();last.legend(handles,labels,loc='upper left',frameon=False)
                last.text(.02,.62,'Positive = better than frozen baseline\nNegative = worse\n\nPhysical RMSE; no dynamic loss weights\nTime excludes queue, validation and saving\nRaw checkpoints / validation points; no smoothing\n\n'+('All 37 pressure levels, equal level mean' if band=='all_levels' else 'Levels >30 hPa, equal level mean'),transform=last.transAxes,va='top',fontsize=10)
                caption='8 selection-validation origins' if scope=='short' else '16 origins disjoint from checkpoint selection'
                fig.suptitle(f'{hw}: {lead}h physical forecast improvement vs training time\n{caption}',fontsize=14)
                name=f'{hw}_{scope}_{lead}h_{band}.png';target=root/'plots'/name;target.parent.mkdir(exist_ok=True)
                temp=target.with_name('.'+name+'.tmp.png');fig.savefig(temp,dpi=150);plt.close(fig);os.replace(temp,target);plots.append(name)
    # Counts at a common saved validation step for fair within-hardware short comparisons.
    latest=[]
    for hw in hardware:
        for scope,lead in [('short',12),('long',120)]:
            for band in ('all_levels','weather_gt30hpa'):
                selected=[r for r in records if r['hardware']==hw and r['scope']==scope and r['lead_hours']==lead and r['band']==band]
                present=[m for m in COLORS if any(r['mode']==m for r in selected)]
                common=None
                if scope=='short' and len(present)==3:
                    common=max(set.intersection(*[set(r['update'] for r in selected if r['mode']==m) for m in present]))
                for mode in present:
                    rows=[r for r in selected if r['mode']==mode and (common is None or r['update']==common)]
                    latest.append(max(rows,key=lambda r:r['update']))
    write(root/'latest_summary.json',dict(rows=latest,failed_checkpoints=failures,generated_at=time.time(),threshold_percent=plan['threshold_percent']))
    lines=['# 全变量物理评估随训练时间的变化','',
           '这是训练进行中的结果快照。训练继续，报告由采样器和GPU评估作业自动更新。','',
           '**改善率 = 100 × (frozen baseline RMSE − checkpoint RMSE) / frozen baseline RMSE。** 正值改善，负值变差。所有误差直接对ERA5计算，不使用训练loss权重。','',
           '覆盖全部7个直接对ERA5验证的输出：温度、位势、纬向风、经向风、比湿、云冰、云液水。每个变量的37个气压层均保留；模型内部native表示不是额外的独立观测变量，不混入这7项计数。','',
           '空间误差使用球面面积权重，跨起点平均MSE后开根号；变量总览再均匀合并所选气压层的MSE。提供全37层和>30hPa两个明确口径，避免只展示有利气压层。','',
           '横轴按本次用户要求使用累计training-step计算时间（小时），不含排队、步骤外初始化、验证和保存。CSV同时保留optimizer step。三组起始时间不同时，也各自从训练时间0计。','',
           '改善/变差计数使用±0.1%的显示容差，之间记为基本不变；不是统计显著性判据。baseline近零时标为无法定义，不强行计算巨大百分比。所有原始百分比仍保留。','',
           '## 最新计数','',
           '短时表按同一hardware三组共同拥有的最新step对齐；五天表按各自最新已评估checkpoint，step明确列出。','',
           '| GPU | 预测 | 层范围 | 模式 | step | 变量 改善/变差/近似不变 | 变量×层 改善/变差/近似不变/未定义 |',
           '| --- | --- | --- | --- | ---: | --- | --- |']
    for r in latest:
        v=r['variable_counts'];n=r['level_counts']
        lines.append(f"| {r['hardware']} | {r['lead_hours']}h ({r['scope']}) | {r['band']} | {r['mode']} | {r['update']} | {v['improved']}/{v['worse']}/{v['unchanged']} | {n['improved']}/{n['worse']}/{n['unchanged']}/{n['undefined']} |")
    lines+=['','## 每个变量的真实改善曲线','']
    for name in plots:lines+=['!['+name+'](plots/'+name+')','']
    lines+=['## 证据与限制','',
            '- 短时曲线使用训练每20步已有的真实物理验证，8个固定起点，6h/12h；这些起点也用于checkpoint选择。',
            '- 五天曲线来自不改动训练的checkpoint快照，另用16个固定起点、6–120h闭环前向；起点不参与短时选择，包含先前小试验已经观察过的部分起点。',
            '- full-field和逐层RMSE分别保存在history与results JSON；CSV为每个变量的聚合值。baseline、源码、资源、统计与checkpoint哈希可追溯。',
            '- 历史checkpoint只长期保留last/best。此前被覆盖的checkpoint没有伪造补跑：短时历史来自已保存的真实验证记录，五天时间点从现存best和本次起开始保存的快照取得。',
            '- 单seed与小验证集的计数是描述性结果，不等于显著改善。失败checkpoint单独报告，不删除失败样本再计算指标。',
            '- [逐变量CSV](variable_improvement.csv)、[全部改善记录](improvement_history.json)、[最新计数](latest_summary.json)、[评估计划](plan.json)。','']
    if failures:lines+=['### 未完成或失败的checkpoint评估','',*['- '+x['id']+': '+x['error_type']+' — '+x['message'] for x in failures],'']
    atomic(root/'REPORT.md','\n'.join(lines).encode())
    print(json.dumps(dict(report=str(root/'REPORT.md'),records=len(records),plots=len(plots),failed_checkpoints=len(failures))),flush=True)

if __name__=='__main__':main()
