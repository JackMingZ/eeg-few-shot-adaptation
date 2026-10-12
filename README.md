# Cross-subject EEG few-shot adaptation

新用户通常需要提供标注脑电才能校准运动想象分类器。本项目研究：每类只有 1、3、5 或 10 条标注试次时，微调 EEGNet 分类头能带来多少改善，是否值得更新整个网络？

Experiments on nine BNCI2014-001 participants compare head-only fine-tuning, full-model fine-tuning and channel-wise adaptation. Head adaptation improves mean accuracy with ten labeled trials per class, but hurts it with only one. These are exploratory results, not a new architecture or a strict reproduction of EEGNet.

## 结果

| 方法 | 每类校准数 | Accuracy | 被试间 SD | 适配参数 | 适配时间（秒） |
|---|---:|---:|---:|---:|---:|
| EEGNet，无适配 | 0 | 41.74% | 16.94% | 0 | — |
| 分类头微调 | 10 | 47.69% | 15.62% | 516 | 0.071 |
| 全模型微调 | 10 | 47.44% | 16.23% | 1,716 | 0.121 |
| 通道缩放 + 分类头 | 10 | 48.50% | 15.74% | 538 | 0.124 |
| 通道缩放/偏移 + 分类头 | 10 | 48.80% | 15.85% | 560 | 0.125 |

主配置为 128 Hz、22 × 256 点、时间核 32、Dropout 0.25，总参数 1,760。每名被试先平均三个种子，再统计九名被试的均值与标准差。时间来自原 RTX 5070 实验，仅反映本机测量。

分类头在 8/9 名被试上改善，平均提高 5.94 个百分点；每类只有 1 条时反而平均下降 2.01 个百分点。仿射方案平均分略高，但方法间比较经 Holm 校正后不足以支持稳定优势。分类头是本次 k=10 实测较快的适配方案，不能由参数量直接推断速度。

[公开结果与重算说明](reports/paper/README.md)包含逐运行指标、逐被试结果、成本、配置和统计表。不提供原始 EEG 或权重；公开表可复算汇总，逐试次审计仍需要本地数据与训练产物。

## 实验划分

每轮一名目标被试，其 T 会话提供校准样本，E 会话固定用于测试。另一名被试提供源域验证，其余七名用于训练；源被试使用两个会话。各适配方法共享源模型和校准样本，预算采用嵌套抽样。适配时冻结 BatchNorm 运行统计并关闭 Dropout；full 更新原网络权重。

这是离线四分类实验，使用 4–38 Hz 滤波、提示后 0.5–2.5 秒窗口，保留伪迹试次。已查看过目标测试结果并作过探索性选择，因此不称为预注册或独立确认实验。

## 快速开始

在仓库根目录运行，以下为 Windows PowerShell 命令。已验证的本地 Python 版本为 3.14；其他系统尚未独立验证。GPU 安装见运行指南。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
# 三被试、一个种子的流程检查，不用于估计模型效果；不生成图片
.\.venv\Scripts\python.exe run_real_pipeline.py --profile paper --subjects 1 2 3 --targets 1 --seeds 7 --epochs 5 --adapt-epochs 5 --budgets 1 3 --output results/real/paper_smoke --no-plots
```

入口会下载对应被试的数据并预处理。完整矩阵最多下载约 780 MB，需要网络；CPU 可运行但完整实验可能较慢。

```powershell
# 完整九被试、三个种子；输出到新目录，不覆盖历史结果
.\.venv\Scripts\python.exe run_real_pipeline.py --profile paper --output results/real/paper_rerun --no-plots
```

默认入口使用 `paper` 配置。`--profile legacy` 保留 125 Hz、500 点的早期实验；底层单次脚本的历史默认值没有更改，不应直接用于重算上表。使用 `--dry-run` 可查看实际命令。

## 限制

- 只有一个九被试数据集，结果不能代表实际在线部署表现。
- 两个源检查点出现单类别预测，已保留，尚需诊断训练稳定性。
- 划分、训练轮数和伪迹政策与原论文不同，不是论文数值的严格复现；小样本统计仅供探索。

## 文档与来源

- [运行指南](REAL_DATA_GUIDE.md)
- [实验报告与公开证据](reports/paper/README.md)
- [研究记录](STUDY_LOG.md)

参考 Lawhern et al. (2018), [EEGNet](https://doi.org/10.1088/1741-2552/aace8c)。数据为 BNCI2014-001 / BCI Competition IV 2a。论文、数据和第三方代码按各自许可使用，项目 MIT 许可不覆盖第三方材料。

项目开发使用了 AI 辅助代码修改、文档整理和检查命令编写。
