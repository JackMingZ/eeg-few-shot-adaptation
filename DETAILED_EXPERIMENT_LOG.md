# 详细实验记录：BCI Competition IV 2a 跨被试少样本适配

> 生成时间：2026-10-10T12:31:57+08:00。本文档由项目中保存的 JSON、CSV、NPZ 元数据和审计结果自动生成，数字不手工改写。
> 实验性质：真实公开 EEG 数据上的探索性实验，不是 EEGNet 论文的严格复现，也不构成确证性统计结论。

## 1. 本轮实验目的

评估 EEGNet 风格轻量网络在新被试上的直接迁移表现，以及每类少量有标签校准样本对不同适配策略的影响。每个目标被试单独适配，适配方法从同一源模型检查点复制权重。

## 2. 运行环境

- 操作系统：Windows-11-10.0.26200-SP0。
- Python：`3.14.4 (tags/v3.14.4:23116f9, Apr  7 2026, 14:10:54) [MSC v.1944 64 bit (AMD64)]`；解释器：`<project-root>/.venv/Scripts/python`。
- PyTorch：`2.11.0+cu128`；CUDA runtime：`12.8`；CUDA 可用：`True`。
- GPU：`NVIDIA GeForce RTX 5070`；显存：11.94 GiB；CUDA 矩阵检查：`True`。
- 关键包：MNE 1.13.2，MOABB 1.7.2，NumPy 2.5.3，SciPy 1.18.1，scikit-learn 1.9.1。
- 环境记录文件：[environment_verified.json](environment_verified.json)。

## 3. 数据来源与完整性

- 数据集：BNCI2014-001 / BCI Competition IV 2a；官方说明：https://lampx.tugraz.at/~bci/database/001-2014/。
- 被试：9 名（1–9），每名包含 `0train` 和 `1test` 两个会话；每名缓存 576 条试次，共 **5184 条**。
- 原始 MAT 文件：18 个（每被试 AxxT.mat、AxxE.mat）；下载脚本保留文件大小、SHA-256 和来源说明。
- 原始 MAT 文件合计字节：779,873,919；缓存 NPZ 合计字节：211,706,762。
- 本地哈希是可追溯校验值；其中原始文件哈希已与 NEMAR sourcedata manifest 记录核对，不能解释为上游签名。

### 3.1 缓存统计

| subject | NPZ shape | label counts (0/1/2/3) | cache SHA-256 (first 16) | bytes |
|---:|---|---|---|---:|
| 1 | `[576, 22, 500]` | `[144, 144, 144, 144]` | `dec04acad1023720…` | 23,530,307 |
| 2 | `[576, 22, 500]` | `[144, 144, 144, 144]` | `0883d99cf4d493d4…` | 23,521,800 |
| 3 | `[576, 22, 500]` | `[144, 144, 144, 144]` | `7d4de9ac94fc5f27…` | 23,528,156 |
| 4 | `[576, 22, 500]` | `[144, 144, 144, 144]` | `ee77c65d6fa514c4…` | 23,526,780 |
| 5 | `[576, 22, 500]` | `[144, 144, 144, 144]` | `b2636cf52f7169f1…` | 23,524,606 |
| 6 | `[576, 22, 500]` | `[144, 144, 144, 144]` | `c1f9bd793af12551…` | 23,530,786 |
| 7 | `[576, 22, 500]` | `[144, 144, 144, 144]` | `d72b3c9139f60d99…` | 23,528,862 |
| 8 | `[576, 22, 500]` | `[144, 144, 144, 144]` | `a6a1a1d1b8eac090…` | 23,511,757 |
| 9 | `[576, 22, 500]` | `[144, 144, 144, 144]` | `17f3dd9bb8b34d3c…` | 23,503,708 |

## 4. 预处理协议

1. 读取 MOABB `BNCI2014_001`，只保留 22 个 EEG 通道，排除 EOG。
2. 每个 recording/run 独立进行 4–38 Hz 零相位 FIR 滤波。
3. 以运动想象提示为事件起点，截取 `[2, 6)` 秒，重采样到 125 Hz。
4. 每条输入为 `22 × 500`，保留微伏单位；训练输入缩放为除以 1.0。
5. 保留所有试次，包括数据提供方标记的伪迹；没有按标签进行剔除。
6. 四类标签为左手、右手、脚、舌头，内部编码 0–3。

## 5. 划分和防泄漏协议

- 每轮只选择一个目标被试；目标被试第一会话中的样本按类别抽取校准集，第二会话作为固定最终测试集。
- 其余 8 名被试中，7 名用于源训练，1 名用于源验证；验证被试按轮次轮换。
- 校准预算为每类 1、3、5、10 条；预算使用嵌套抽样，类别保持平衡。
- 最终测试集不用于训练、适配、模型选择或归一化统计估计。
- 9 个目标被试 × 3 个随机种子 = 27 个源训练组合；每个组合产生源模型、CSP+LDA 和 4 种适配方法的指标。

## 6. 方法、参数和训练配置

| method | updated parameters | trainable parameters | note |
|---|---|---:|---|
| source | no adaptation | 0 | direct source-model test |
| CSP + LDA | CSP features and LDA | — | source-only traditional baseline |
| head | classifier head | 964 | freeze EEGNet backbone |
| scale_head | per-channel scale + head | 986 | update scale and classifier head |
| affine_head | per-channel scale/shift + head | 1008 | input affine parameters and head |
| full | full network | 2420 | freeze BatchNorm statistics and disable Dropout during adaptation |
- EEGNet 风格网络总参数：2464；输入 22 通道、500 时间点、4 类输出。
- 源训练：最多 100 轮，早停 patience=20；适配最多 50 轮，学习率 0.001；设备 `cuda`。
- 随机种子：[7, 17, 27]；输入缩放：1.0；结果目录：`results/real/nine_subject_exploratory`。

## 7. 可重复运行入口

```powershell
.\.venv\Scripts\python.exe verify_environment.py
.\.venv\Scripts\python.exe prepare_real_data.py --subjects 1 2 3 4 5 6 7 8 9
.\.venv\Scripts\python.exe run_real_batch.py --targets 1 2 3 4 5 6 7 8 9 --seeds 7 17 27 --epochs 100 --input-scale 1.0 --output results/real/nine_subject_exploratory
.\.venv\Scripts\python.exe audit_real_results.py --root results/real/nine_subject_exploratory
.\.venv\Scripts\python.exe summarize_real_results.py --root results/real/nine_subject_exploratory
```

本次记录使用的主要代码快照为每个 `target_xx_seed_yy/config.json` 中的 `code_sha256`；每轮目录也保存了 `code/` 快照、划分、检查点、预测、损失和日志。

## 8. 汇总结果

下表来自 `summary.csv`，均为 9 个目标被试的被试级均值；标准差是被试间标准差，3 个种子先在同一被试内取均值，因此没有把种子当成独立被试。

| method | calibration per class | Accuracy mean +/- SD | Macro-F1 mean +/- SD | Kappa mean +/- SD |
|---|---:|---:|---:|---:|
| affine_head | 1 | 40.19% ± 13.27% | 37.67% ± 14.53% | 20.25% ± 17.70% |
| affine_head | 3 | 44.17% ± 15.35% | 42.57% ± 15.81% | 25.57% ± 20.47% |
| affine_head | 5 | 46.24% ± 15.22% | 44.65% ± 15.63% | 28.33% ± 20.29% |
| affine_head | 10 | 48.56% ± 15.73% | 46.66% ± 16.19% | 31.41% ± 20.97% |
| csp_lda | 0 | 35.46% ± 12.89% | 29.77% ± 15.40% | 13.94% ± 17.19% |
| full | 1 | 39.87% ± 13.26% | 37.21% ± 15.32% | 19.82% ± 17.68% |
| full | 3 | 43.22% ± 13.82% | 41.38% ± 14.64% | 24.30% ± 18.43% |
| full | 5 | 43.56% ± 15.13% | 42.18% ± 15.84% | 24.74% ± 20.18% |
| full | 10 | 47.06% ± 16.55% | 45.85% ± 17.05% | 29.41% ± 22.07% |
| head | 1 | 39.90% ± 13.09% | 37.18% ± 14.16% | 19.87% ± 17.46% |
| head | 3 | 43.38% ± 14.27% | 41.52% ± 15.19% | 24.50% ± 19.02% |
| head | 5 | 45.51% ± 15.41% | 43.58% ± 16.10% | 27.35% ± 20.54% |
| head | 10 | 48.44% ± 14.85% | 46.64% ± 15.18% | 31.26% ± 19.79% |
| scale_head | 1 | 40.08% ± 13.62% | 37.52% ± 14.87% | 20.11% ± 18.17% |
| scale_head | 3 | 44.07% ± 15.14% | 42.48% ± 15.60% | 25.43% ± 20.18% |
| scale_head | 5 | 46.36% ± 15.28% | 44.71% ± 15.76% | 28.48% ± 20.37% |
| scale_head | 10 | 48.62% ± 15.75% | 46.73% ± 16.21% | 31.50% ± 21.00% |
| source | 0 | 43.88% ± 16.93% | 38.23% ± 20.58% | 25.17% ± 22.57% |

## 9. 每个目标被试的 k=10 结果

| 目标被试 | source | head | full | affine_head | scale_head | CSP+LDA |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 65.39% | 62.27% | 60.76% | 61.34% | 61.92% | 26.74% |
| 2 | 36.11% | 34.84% | 28.82% | 32.06% | 31.94% | 30.21% |
| 3 | 66.55% | 71.30% | 69.44% | 73.61% | 73.73% | 60.42% |
| 4 | 27.55% | 39.12% | 33.10% | 38.77% | 39.24% | 34.03% |
| 5 | 26.62% | 27.66% | 25.81% | 27.20% | 27.89% | 27.08% |
| 6 | 29.05% | 37.73% | 38.66% | 37.50% | 37.27% | 25.00% |
| 7 | 31.37% | 46.18% | 44.56% | 48.38% | 47.22% | 27.08% |
| 8 | 53.47% | 54.86% | 56.25% | 55.90% | 55.56% | 34.38% |
| 9 | 58.80% | 62.04% | 66.09% | 62.27% | 62.85% | 54.17% |

## 10. 相对源模型的变化

- `paired_gains.csv` 保存每一次运行相对于同一目标、同一种子源模型的 Accuracy 差值。
- 每类 1 条校准时，适配结果可能低于源模型，说明负迁移存在；不能只报告平均提升。
- 每类 10 条时，head、full、affine_head、scale_head 的总体均值分别为 48.44%、47.06%、48.56%、48.62%；相对源模型的差值必须结合逐被试表和 `paired_gains.csv` 解读。
- affine_head 相对 head 的平均差约 +0.12 个百分点；差异很小，当前证据不足以支持其明确优势。

## 11. 时间、显存和参数成本

下表是 `cost_summary.csv` 的 27 轮算术平均。显存为进程内 CUDA 最大已分配内存，包含常驻模型；不是方法独占增量显存。

| method | k | trainable parameters | adaptation seconds | inference seconds | peak CUDA allocation |
|---|---:|---:|---:|---:|---:|
| affine_head | 1 | 1008 | 0.094 | 0.004 | 21.87 MiB |
| affine_head | 3 | 1008 | 0.129 | 0.004 | 30.88 MiB |
| affine_head | 5 | 1008 | 0.166 | 0.005 | 39.90 MiB |
| affine_head | 10 | 1008 | 0.260 | 0.005 | 62.61 MiB |
| csp_lda | 0 | — | — | — | — |
| full | 1 | 2420 | 0.107 | 0.004 | 21.89 MiB |
| full | 3 | 2420 | 0.107 | 0.004 | 30.90 MiB |
| full | 5 | 2420 | 0.126 | 0.004 | 39.92 MiB |
| full | 10 | 2420 | 0.170 | 0.004 | 62.85 MiB |
| head | 1 | 964 | 0.063 | 0.004 | 20.37 MiB |
| head | 3 | 964 | 0.066 | 0.004 | 26.42 MiB |
| head | 5 | 964 | 0.067 | 0.004 | 32.46 MiB |
| head | 10 | 964 | 0.064 | 0.004 | 47.57 MiB |
| scale_head | 1 | 986 | 0.091 | 0.004 | 21.87 MiB |
| scale_head | 3 | 986 | 0.128 | 0.004 | 30.88 MiB |
| scale_head | 5 | 986 | 0.165 | 0.004 | 39.90 MiB |
| scale_head | 10 | 986 | 0.259 | 0.005 | 62.82 MiB |
| source | 0 | 0 | 0.000 | 0.005 | 0.00 MiB |

## 12. 失败批次、诊断和审计

- 早期 `three_subject_pilot` 使用输入除以 100，结果与微伏批次不可混合；该批次保留为失败诊断，不纳入本报告。
- `source_only_diagnostic` 和源训练规模补充实验改变了数据规模或输入设置，只用于诊断，不能替代主实验。
- 主实验 `target_05_seed_7` 的源检查点只预测一个类别；审计保留此警告，说明该组合的迁移基线不可靠。
- 已完成组合：27；审计通过：partition disjointness, nested class-balanced calibration, code snapshots, input cache hashes, shared source checkpoint hashes, prediction IDs and labels, recomputed metrics, confusion matrices, CSP checkpoint reload。
- 审计警告：1 条；详见 audit.json（本地生成文件：`audit.json`）。
- 五项协议单元测试、CUDA 运算、pip check 和 Python 编译检查已运行；测试输出应以本次终端记录为准。

## 13. 论文依据与实现边界

项目灵感来自 Lawhern et al. (2018) EEGNet：时间卷积提取时间/频率信息，深度空间卷积学习通道关系，并强调紧凑模型。当前网络是 EEGNet 风格简化实现，不是原论文架构、训练协议或结果的逐项复现。
适配方法是本项目的实验性比较，不能表述为首次提出；相关方法核对见 [references/LITERATURE_NOTES.md](references/LITERATURE_NOTES.md)。

## 14. 局限与下一步

1. 当前只有一个公开数据集和固定预处理；需要独立数据集或不同预处理验证稳定性。
2. 预算、学习率和适配轮数仍是预设探索配置；正式确认前应使用不触碰最终测试集的开发集进行选参。
3. 需要补充统计检验、置信区间和更严格的预注册式比较，避免把小幅差异解释为方法优势。
4. 可继续报告逐被试混淆矩阵、训练曲线和检查点重载结果，并保留本目录中的原始预测作为审计依据。

## 15. 文件索引

- REPORT.md（本地生成文件：`REPORT.md`）：自动汇总报告。
- all_metrics.csv（本地生成文件：`all_metrics.csv`）：逐目标、逐种子、逐方法指标。
- subject_means.csv（本地生成文件：`subject_means.csv`）：逐被试均值。
- paired_gains.csv（本地生成文件：`paired_gains.csv`）：相对同源模型的逐运行变化。
- cost_summary.csv（本地生成文件：`cost_summary.csv`）：参数、时间和显存成本。
- audit.json（本地生成文件：`audit.json`）：数据划分、哈希、预测和指标审计。
- `target_XX_seed_YY/`：每轮配置、代码快照、划分、模型、预测、损失和日志。
