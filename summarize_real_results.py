"""Summarize completed real runs without treating seeds as independent subjects."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def summarize(root):
    root=Path(root)
    records=[]
    signatures=set()
    collapsed=0
    fields=['subjects','epochs','patience','adapt_epochs','adapt_lr','input_scale_microvolt','budgets',
            'data_root','samples','sampling_rate','epoch_start','epoch_end','temporal_kernel','dropout','euclidean_alignment']
    for path in sorted(root.glob('target_*/metrics.json')):
        cfg=json.loads((path.parent/'config.json').read_text(encoding='utf-8'))
        defaults={'data_root':'data/processed','samples':500,'sampling_rate':125.,'epoch_start':2.,'epoch_end':6.,'temporal_kernel':64,'dropout':0.5,'euclidean_alignment':False}
        signatures.add(json.dumps({key:cfg.get(key, defaults.get(key)) for key in fields},sort_keys=True))
        with np.load(path.parent/'source_k0_predictions.npz',allow_pickle=False) as prediction:
            collapsed+=int(len(np.unique(prediction['y_pred']))==1)
        records.extend(json.loads(path.read_text(encoding='utf-8')))
    if not records: raise ValueError('No completed experiments')
    if len(signatures)!=1: raise ValueError('Do not pool runs with different protocols/configurations')
    frame=pd.DataFrame(records)
    frame.to_csv(root/'all_metrics.csv',index=False)
    per_subject=frame.groupby(['target','method','budget_per_class'],as_index=False)[['accuracy','macro_f1','kappa']].mean()
    per_subject.to_csv(root/'subject_means.csv',index=False)
    summary=per_subject.groupby(['method','budget_per_class'])[['accuracy','macro_f1','kappa']].agg(['mean','std','count'])
    summary.to_csv(root/'summary.csv')
    source=frame[frame.method=='source'][['target','seed','accuracy']].rename(columns={'accuracy':'source_accuracy'})
    gains=frame.merge(source,on=['target','seed'],how='left')
    gains['accuracy_gain']=gains.accuracy-gains.source_accuracy
    gains.to_csv(root/'paired_gains.csv',index=False)
    cost_columns=['parameters','trainable_parameters','adaptation_seconds','source_training_seconds','inference_seconds','peak_cuda_bytes']
    costs=frame.groupby(['method','budget_per_class'])[cost_columns].mean()
    costs.to_csv(root/'cost_summary.csv')
    fig,ax=plt.subplots(figsize=(8,5))
    for method in ['head','full','affine_head','scale_head']:
        part=per_subject[per_subject.method==method].groupby('budget_per_class').accuracy.agg(['mean','std'])
        if len(part): ax.errorbar(part.index,part['mean'],yerr=part['std'].fillna(0),marker='o',capsize=3,label=method)
    for method,color in [('source','black'),('csp_lda','gray')]:
        part=per_subject[per_subject.method==method]
        if len(part): ax.axhline(part.accuracy.mean(),linestyle='--',color=color,label=method)
    ax.set_xticks(sorted(set(frame.loc[frame.budget_per_class>0,'budget_per_class'])))
    ax.set(xlabel='Labeled trials per class',ylabel='Accuracy',ylim=(0,1),title='Real EEG exploratory results (subject means; SD across subjects)')
    ax.legend(); fig.tight_layout(); fig.savefig(root/'budget_accuracy.png',dpi=160); plt.close(fig)
    for directory in sorted(root.glob('target_*')):
        history=directory/'training_history.json'
        if history.exists():
            h=pd.DataFrame(json.loads(history.read_text(encoding='utf-8')))
            fig,ax=plt.subplots(figsize=(6,4))
            ax.plot(h.epoch,h.train_loss,label='train'); ax.plot(h.epoch,h.validation_loss,label='validation')
            ax.set(xlabel='Epoch',ylabel='Cross-entropy'); ax.legend(); fig.tight_layout()
            fig.savefig(directory/'training_curve.png',dpi=120); plt.close(fig)
        for prediction in directory.glob('*_predictions.npz'):
            with np.load(prediction,allow_pickle=False) as d:
                cm=d['confusion_matrix']
            fig,ax=plt.subplots(figsize=(4,4)); ax.imshow(cm,cmap='Blues')
            for (i,j),value in np.ndenumerate(cm): ax.text(j,i,str(value),ha='center',va='center')
            ax.set(xlabel='Predicted',ylabel='True',xticks=range(4),yticks=range(4),title=prediction.stem.replace('_predictions',''))
            fig.tight_layout(); fig.savefig(prediction.with_suffix('.png'),dpi=100); plt.close(fig)
    lines=['# 真实 EEG 探索性实验报告','',
           f'已完成 {frame[["target","seed"]].drop_duplicates().shape[0]} 个被试/种子组合，{frame.target.nunique()} 个目标被试。',
           '本报告由原始 metrics.json 自动生成；未完成的组合不计入。当前结果是预设设置下的探索性实验，不是论文复现结果或确证性统计结论。','',
           f'每轮源训练使用 {len(cfg["subjects"])-2} 个被试，1 个源被试验证，1 个目标被试测试。模型输入为微伏除以 {cfg["input_scale_microvolt"]:g}。源模型仅输出一个类别的组合数：{collapsed}；若大于零，对应基线的迁移能力不足，不能只看适配后的相对提升。','',
           f'协议：目标第一会话选校准样本、第二会话固定测试；另一个源被试用于验证，其余源被试训练。保持带伪迹试次。输入为 22 通道、{cfg.get("sampling_rate", 125):g} Hz，试次开始后 [{cfg.get("epoch_start", 2):g}, {cfg.get("epoch_end", 6):g}) 秒，4–38 Hz 滤波。',
           '方法：EEGNet-8,2 风格 PyTorch 实现，所有适配从同一源权重出发。head 更新分类头；full 更新原网络全部参数但固定 BN 统计；affine_head 更新通道缩放、偏移和分类头；scale_head 只更新缩放与分类头。CSP+LDA 仅源训练。','',
           '|方法|每类校准数|Accuracy 均值|被试间标准差|Macro-F1 均值|Kappa 均值|',
           '|---|---:|---:|---:|---:|---:|']
    for (method,budget),row in summary.iterrows():
        sd=row[('accuracy','std')]
        spread=f'{sd:.4f}' if np.isfinite(sd) else '不适用（仅一被试）'
        lines.append(f'|{method}|{budget}|{row[("accuracy","mean")]:.4f}|{spread}|{row[("macro_f1","mean")]:.4f}|{row[("kappa","mean")]:.4f}|')
    lines += ['', '## 适配成本（各运行算术平均）', '',
              '|方法|每类校准数|适配参数数|适配秒数|测试推理秒数|',
              '|---|---:|---:|---:|---:|']
    for (method,budget),row in costs.iterrows():
        if method=='csp_lda': continue
        lines.append(f'|{method}|{budget}|{row.trainable_parameters:.0f}|{row.adaptation_seconds:.3f}|{row.inference_seconds:.3f}|')
    if all((method,10) in summary.index for method in ['head','full','affine_head']):
        head=summary.loc[('head',10),('accuracy','mean')]
        affine=summary.loc[('affine_head',10),('accuracy','mean')]
        full=summary.loc[('full',10),('accuracy','mean')]
        lines += ['', '## 预设每类 10 条条件的直接比较', '',
                  f'全模型 {full:.2%}，仅分类头 {head:.2%}，通道仿射加分类头 {affine:.2%}；仿射方法相对分类头的平均差为 {(affine-head)*100:+.2f} 个百分点。此为描述性比较，未据此做显著性或普适性声明。',
                  '通道仿射参数仅有 44 个，但还需更新 964 个分类头参数；且梯度必须穿过主干传播到输入。因此更新参数量减少不必然带来时间或显存节省，成本应以本次实测为准。']
    lines += ['', '## 解释与局限','',
              '先对同一被试的种子取平均，再汇总被试均值与标准差；未将种子当作独立被试。校准收益和下降见 paired_gains.csv。完整配置、试次划分、源权重摘要、训练曲线及预测保存在各 target 目录。',
           '适配轮数和学习率为预设探索设置，尚未做源域模拟适配验证选参。当前全模型微调固定 BatchNorm 统计并关闭 Dropout，需与其他微调策略区分。尚无独立数据集验证，不能声称通道适配新颖或普遍优于全模型。',
              '时间为本机测量；推理时间包含数据传输和批处理开销。显存为进程内 CUDA 最大已分配内存，包含常驻模型，不能解释为方法独占的增量显存。',
              '', '![预算曲线](budget_accuracy.png)']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Summarized {len(frame)} rows into {root}')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',default='results/real/pilot')
    summarize(p.parse_args().root)
