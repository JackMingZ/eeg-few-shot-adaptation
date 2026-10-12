# 128 Hz 主配置：公开实验结果

本目录从本地 `results/real/paper_reproduction/target_*/metrics.json` 导出 27 个运行、486 条记录。它对应首页表格；125 Hz 早期探索没有混入。所有结果均为探索性分析。

## 文件

- [all_metrics.csv](all_metrics.csv)：每名被试、每个种子、每种方法和预算的统计输入，含实测成本。
- [subject_means.csv](subject_means.csv)：每名被试先平均三个种子。
- [summary.csv](summary.csv)：九名被试的均值、样本标准差和数量。
- [cost_summary.csv](cost_summary.csv)：方法/预算的成本均值。
- [configuration.json](configuration.json)：原实验批量配置，路径表示本地运行位置。
- [provenance.json](provenance.json)：逐运行配置、代码/缓存/指标 SHA256。哈希辅助对应产物，不证明科学有效性。
- [paired_subject_statistics.csv](paired_subject_statistics.csv)：适配相对 source 的被试级 bootstrap 95% 区间、未校正 Wilcoxon p。
- [paired_method_comparisons.csv](paired_method_comparisons.csv)：4 个预算 × 6 对方法，24 次检验统一 Holm 校正。

不含 EEG、权重和逐试次预测。外部读者可以从公开指标重算汇总和统计，但不能仅凭这些表重做逐试次审计。原始训练配置和完整数据仍是从头复现的必要条件。

## 结果解释

head 每类 10 条时平均提高 5.94 个百分点，8/9 名被试改善，bootstrap 区间约 [2.71, 9.25] 个百分点；每类 1 条时平均下降 2.01 个百分点。置信区间使用被试为抽样单位，三个种子不是三个独立被试。

affine_head 在 k=10 比 head 高约 1.12 个百分点；对应方法比较 Holm p=0.375。当前不能据此证明稳定优于 head，也不能把未显著解释为等效。未校正的适配对 source 检验仅用于探索，不能挑其中显著项作确认性声明。

原矩阵 `target_04_seed_17` 和 `target_04_seed_27` 的源模型只输出一个类别，保留全部运行。时间来自 RTX 5070，本机短时计时存在波动；显存是进程峰值，不能视为方法独占显存。

## 重算

在仓库根目录、安装依赖后运行（不需要原始 EEG，不生成图片）：

```powershell
python analyze_subject_gains.py --root reports/paper --seed 20261010 --bootstrap-samples 20000
python analyze_subject_gains.py --root reports/paper --method-comparisons --seed 20261011 --bootstrap-samples 20000
```

均值表按 `all_metrics.csv` 的 `target, method, budget_per_class` 分组求均值，再跨被试统计；成本按 `method, budget_per_class` 分组。

如本地已有原始实验产物，可以重新导出：

```powershell
python export_public_results.py --source results/real/paper_reproduction --output reports/paper
```

这会替换本目录汇总表；更新后应重新运行上述统计命令，并核对首页。导出程序核对九被试 × 种子 7/17/27 × 完整方法/预算集合，拒绝缺项、重复项或额外项。同时检查逐运行身份、主输入配置、100 轮训练、patience=20、50 次适配、学习率 0.001、输入比例、预算与批量计划一致，所有检查在写出之前完成。

这些表由已有指标重算，本次未重新训练完整矩阵；历史数值的逐试次审计记录与当前重新计算的汇总统计是不同层次的证据。
