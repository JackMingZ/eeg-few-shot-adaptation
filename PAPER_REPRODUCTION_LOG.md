# EEGNet 论文配置分支记录

## 目的

这一分支用于先建立可信的 EEGNet 基线，再决定是否继续少样本适配。它不与主实验 `results/real/nine_subject_exploratory` 混合汇总。

## 已实现的论文式配置

- 采样率：128 Hz；
- 事件后窗口：试次开始后 2.5–4.5 秒，即提示后 0.5–2.5 秒；
- 每条输入：22 × 256；
- 时间卷积核：32；
- Dropout：0.25；
- 其余网络保持 EEGNet 风格结构；
- 数据仍来自 BNCI2014-001，保留四类运动想象和所有试次；
- 目标测试集仍固定为目标被试 E 会话，校准使用目标 T 会话。

## 首个流程验证

运行目录：`results/real/paper_reproduction_pilot/target_01_seed_7`。

命令：

```powershell
.\.venv\Scripts\python.exe prepare_real_data.py --output-dir processed_paper --sampling-rate 128 --epoch-start 2.5 --epoch-end 4.5
.\.venv\Scripts\python.exe run_real_experiment.py --data-root data/processed_paper --samples 256 --sampling-rate 128 --epoch-start 2.5 --epoch-end 4.5 --temporal-kernel 32 --dropout 0.25 --target 1 --seed 7 --output results/real/paper_reproduction_pilot --epochs 100 --adapt-epochs 50 --budgets 1 3 5 10
```

首个目标被试结果：

| 方法 | k=0/10 | Accuracy |
|---|---:|---:|
| EEGNet source | 0 | 62.15% |
| 分类头微调 | 10 | 62.15% |
| 全模型微调 | 10 | 62.85% |
| scale + head | 10 | 63.89% |
| affine + head | 10 | 61.81% |
| CSP + LDA | 0 | 37.15% |

这只是一个目标被试、一个随机种子，不能作为论文复现结论，也不能证明任何适配方法优于其他方法。它只证明论文式窗口、采样率、模型参数和保存/重载流程已经可以运行。

## 九被试矩阵结果

随后已完成 9 个目标被试 × 3 个随机种子的完整矩阵，共 486 条指标记录。审计通过划分无交集、嵌套校准、缓存哈希、代码快照、同源检查点、预测重算、混淆矩阵和 CSP 重载检查。结果目录为 `results/real/paper_reproduction`，摘要见其中的 `REPORT.md`、`summary.csv` 和 `cost_summary.csv`。

| 方法 | k | Accuracy 均值 | 被试间 SD |
|---|---:|---:|---:|
| EEGNet source | 0 | 41.74% | 16.94% |
| CSP + LDA | 0 | 38.43% | 13.43% |
| 分类头微调 | 10 | 47.69% | 15.62% |
| 全模型微调 | 10 | 47.44% | 16.23% |
| 仅缩放 + 分类头 | 10 | 48.50% | 15.74% |
| 缩放/偏移 + 分类头 | 10 | 48.80% | 15.85% |

在这套配置下，k=10 时仿射方案相对分类头约高 1.12 个百分点，但仍只有一次完整矩阵，且通道仿射适配成本高于分类头。这个差异值得作为后续统计检验的候选现象，暂时不能写成稳定优势。

完整矩阵中有两个源模型只预测单一类别，已保留在审计警告中；不删除这些组合，也不根据目标测试表现挑选种子。

## 被试级成对分析

对每个目标被试先平均 3 个种子，再计算适配方法相对同一目标无适配源模型的 Accuracy 差值。结果保存在 `results/real/paper_reproduction/paired_subject_statistics.csv`。

| 方法 | k | 平均提升 | 提升被试数 | bootstrap 95% 区间 | Wilcoxon p（未校正） |
|---|---:|---:|---:|---:|---:|
| head | 1 | -2.01 个百分点 | 5/9 | [-6.02, 1.71] | 0.570 |
| head | 5 | +2.73 个百分点 | 6/9 | [0.05, 5.49] | 0.129 |
| head | 10 | +5.94 个百分点 | 8/9 | [2.71, 9.28] | 0.008 |
| full | 10 | +5.70 个百分点 | 7/9 | [1.58, 10.19] | 0.043 |
| scale_head | 10 | +6.75 个百分点 | 9/9 | [3.45, 10.20] | 0.004 |
| affine_head | 10 | +7.06 个百分点 | 9/9 | [3.65, 10.52] | 0.004 |

这些 p 值没有进行多重比较校正，样本量只有 9 名被试，不能直接写成确认性显著性结论。它们支持继续检查 k=10 的适配效果，但不能单独证明仿射方案优于分类头。

## 与论文的剩余差异

- 当前仍使用项目自己的源/验证轮换，而不是论文中的完整随机重复划分；
- 当前训练上限仍为 100 轮、patience 20，不是论文的完整 500 轮设置；
- 当前仍保留伪迹试次，且使用本项目的离线试次评估；
- 当前适配分支是项目扩展，不属于论文原始实验。

## 后续问题

成对差值、bootstrap/Wilcoxon 及方法间 Holm 校正已完成，公开表见 [主报告](reports/paper/README.md)。EA 已做隔离试验并保留负结果。后续重点是源域模拟目标的选参和新的独立验证，不以现有目标测试结果继续挑选配置。
