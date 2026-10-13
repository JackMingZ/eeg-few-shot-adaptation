# Cross-subject EEG few-shot adaptation

用 EEGNet 做运动想象分类时，换一个被试，模型还能用好吗？这个项目关注其中一个具体问题：给新被试每类 1、3、5 或 10 条标注脑电，只微调分类头，和微调整个网络相比，效果有多大差别？

实验使用 BNCI2014-001（BCI Competition IV 2a）的 9 名被试，比较了分类头微调、全模型微调，以及输入通道缩放和偏移。代码基于 EEGNet 的时空卷积结构，也保留了 CSP+LDA 作为传统方法对照。

## 结果

| 方法 | 每类校准数 | Accuracy | 被试间 SD | 适配参数 | 适配时间（秒） |
|---|---:|---:|---:|---:|---:|
| EEGNet，无适配 | 0 | 41.74% | 16.94% | 0 | — |
| 分类头微调 | 10 | 47.69% | 15.62% | 516 | 0.071 |
| 全模型微调 | 10 | 47.44% | 16.23% | 1,716 | 0.121 |
| 通道缩放 + 分类头 | 10 | 48.50% | 15.74% | 538 | 0.124 |
| 通道缩放/偏移 + 分类头 | 10 | 48.80% | 15.85% | 560 | 0.125 |

表中结果先对每名被试的三个随机种子取平均，再计算九人的均值和标准差。模型输入为 22 通道 × 256 点（128 Hz），时间卷积核长度为 32，Dropout 为 0.25，总参数 1,760。适配时间在 RTX 5070 上测得，换一台机器可能会不同。

每类有 10 条样本时，分类头微调让 8 名被试的准确率提高，平均提升 5.94 个百分点。但样本少到每类只有 1 条时，平均准确率反而下降了 2.01 个百分点。少量校准并不总是有帮助。

通道缩放和偏移的平均分更高，不过在 Holm 多重比较校正后，还不能确定它稳定优于分类头微调。分类头更新的参数少，在这次实验中也更快，因此保留它作为主要参照。

完整指标、逐被试结果和统计脚本说明放在 [reports/paper](reports/paper/README.md)。这些表可以用于重算汇总结果；原始脑电和模型权重没有上传，逐试次核对需要在本地重新运行。

## 实验怎么做

每轮留出一名目标被试，用其 T 会话抽取校准样本，E 会话做测试。其余八人中，一人用于验证，七人用于训练，训练和验证都使用各自的两个会话。

几种适配方法从同一个源模型开始，使用同一批校准样本。较大的校准集包含较小校准集中的样本，减少抽样差异对比较的影响。适配时固定 BatchNorm 运行统计并关闭 Dropout；全模型微调更新原网络权重。

预处理保留 22 个 EEG 通道，做 4–38 Hz 滤波，截取提示后 0.5–2.5 秒，重采样到 128 Hz。任务是离线四分类，数据中带伪迹标记的试次也保留。

## 快速开始

下面的命令在仓库根目录执行，适用于 Windows PowerShell。当前已在 Windows、Python 3.14 上验证，其他系统尚未单独测试；GPU 安装方法见 [运行指南](REAL_DATA_GUIDE.md)。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

先用三名被试跑一个小实验，检查下载、预处理和训练能否正常完成：

```powershell
.\.venv\Scripts\python.exe run_real_pipeline.py `
  --profile paper --subjects 1 2 3 --targets 1 --seeds 7 `
  --epochs 5 --adapt-epochs 5 --budgets 1 3 `
  --output results/real/paper_smoke --no-plots
```

这条命令只训练 5 轮，用来检查流程，不能与上表的准确率直接比较。完整实验使用九名被试、三个随机种子：

```powershell
.\.venv\Scripts\python.exe run_real_pipeline.py --profile paper --output results/real/paper_rerun --no-plots
```

脚本会下载所需数据并预处理，完整数据约 780 MB。没有 GPU 也能运行，但完整训练可能较慢。`--no-plots` 跳过画图，`--dry-run` 只打印各步骤的命令。

上表对应 `paper` 配置。早期的 125 Hz、500 点实验保留在 `--profile legacy` 中，两套结果不要混用。直接调用底层脚本时需要显式指定配置，详见运行指南。

## 还有哪些问题

- 目前只用了一个九被试数据集，还没有验证在线使用的效果。
- 两个源模型只预测一个类别。这些运行没有删掉，训练稳定性仍需进一步检查。
- 开发过程中查看过目标测试结果，并据此做过探索性选择，因此这些结果只能作为探索性分析。
- 数据划分、训练轮数和伪迹处理与 EEGNet 原论文不同，这里并未严格复现论文数值。

## 文档与来源

- [运行指南](REAL_DATA_GUIDE.md)
- [实验报告与公开证据](reports/paper/README.md)
- [研究记录](STUDY_LOG.md)

参考 Lawhern et al. (2018), [EEGNet](https://doi.org/10.1088/1741-2552/aace8c)。数据为 BNCI2014-001 / BCI Competition IV 2a。论文、数据和第三方代码按各自许可使用，项目 MIT 许可不覆盖第三方材料。

项目开发使用了 AI 辅助代码修改、文档整理和检查命令编写。
