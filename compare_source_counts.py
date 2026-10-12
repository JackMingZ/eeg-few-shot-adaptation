"""Supplement: nested source-set sizes for fixed target 1 / validation subject 2."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    records=[];reference={}
    for directory in ['three_subject_microvolt','source_count_three','nine_subject_exploratory']:
        for seed in [7,17,27]:
            run=ROOT/'results/real'/directory/f'target_01_seed_{seed}'
            cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
            splits=json.loads((run/'splits.json').read_text(encoding='utf-8'))
            assert cfg['validation_subject']==2 and cfg['input_scale_microvolt']==1.
            signature={k:splits[k] for k in ['validation','calibration_pool','test','calibration']}
            if seed in reference: assert signature==reference[seed], 'Target/validation partition changed'
            else: reference[seed]=signature
            for row in json.loads((run/'metrics.json').read_text(encoding='utf-8')):
                records.append(row|{'n_source_subjects':len(cfg['subjects'])-2,'result_directory':directory})
    out=ROOT/'results/real/source_count_comparison';out.mkdir(exist_ok=True)
    frame=pd.DataFrame(records);frame.to_csv(out/'raw_metrics.csv',index=False)
    summary=frame.groupby(['n_source_subjects','method','budget_per_class'])[['accuracy','source_training_seconds']].agg(['mean','std'])
    summary.to_csv(out/'summary.csv')
    fig,ax=plt.subplots(figsize=(7,5))
    for method,budget in [('source',0),('head',10),('full',10),('affine_head',10)]:
        subset=frame[(frame.method==method)&(frame.budget_per_class==budget)]
        table=subset.groupby('n_source_subjects').accuracy.agg(['mean','std'])
        ax.errorbar(table.index,table['mean'],yerr=table['std'],marker='o',capsize=3,label=f'{method} (k={budget})')
    ax.set(xlabel='Source training subjects',ylabel='Accuracy',ylim=(0,1),xticks=[1,3,7],
           title='Target 1 only: nested source sets (mean +/- seed SD)')
    ax.legend();fig.tight_layout();fig.savefig(out/'source_count_accuracy.png',dpi=160);plt.close(fig)
    lines=['# 源被试数量补充比较','',
           '固定目标被试1、验证被试2，逐步使用源训练集合 {3}、{3,4,5}、{3,4,5,6,7,8,9}。每种设置使用种子7/17/27，校准池、测试集和验证集一致性已经自动核查。',
           '这是一名目标被试上的嵌套集合示例；源数量、源组成和每轮梯度更新数同时变化，不能把差异解释成普适、纯粹的“被试数量效应”。图中波动是种子标准差，不是被试间标准差。','',
           '|源训练被试数|无适配 Accuracy|全模型，每类10条|仿射+分类头，每类10条|',
           '|---|---:|---:|---:|']
    for count in [1,3,7]:
        values=[summary.loc[(count,m,k),('accuracy','mean')] for m,k in [('source',0),('full',10),('affine_head',10)]]
        lines.append(f'|{count}|{values[0]:.2%}|{values[1]:.2%}|{values[2]:.2%}|')
    lines+=['','![源数量比较](source_count_accuracy.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:-2]))
