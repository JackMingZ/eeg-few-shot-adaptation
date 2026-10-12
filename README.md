# 跨被试 EEG 少样本适配

这是一个面向脑机接口研究申请的可复现实验工程：在真实运动想象 EEG 上研究新被试的少样本适配，并同时考察准确率、参数量、适配时间和负迁移。若想先了解项目动机、本人完成的工作和下一步计划，参见 [申请项目说明](APPLICATION_PROJECT.md)。独立环境验证记录见 [REPRODUCIBILITY_SMOKE_TEST.md](REPRODUCIBILITY_SMOKE_TEST.md)，配图设计稿见 [FIGURE_SPEC.md](FIGURE_SPEC.md)。

以 EEGNet 时空特征提取结构为基线，比较真实运动想象 EEG 上的分类头微调、全模型微调、通道缩放/偏移加分类头适配，并分析参数与时间成本。当前真实模型的参数统计见 [模型概况](MODEL_PROFILE.md)。

**2026-10-08：九被试×三种子的真实数据探索实验已完成，共486行指标。** [中文研究报告](RESEARCH_REPORT.md) · [结果索引](RESULTS_INDEX.md) · [研究记录](STUDY_LOG.md)。每类10条时 head 48.44%、full 47.06%、affine_head 48.56%；尚无明确证据支持通道仿射优于简单分类头。不能将探索结果解释为严格复现或确证性结论。

当前已有真实数据下载、预处理、GPU 训练、适配、评估、批量实验、报告和离线预测入口。实际完成的实验只认 `results/real/` 下的 `metrics.json`；尚未产生的结果不能视为已完成。原来的合成演示保留，两个流程互相独立。

## 真实数据运行

在项目目录使用已配置的虚拟环境：

```powershell
.\.venv\Scripts\python.exe verify_environment.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
# 完整流程：下载、预处理、9 被试 × 3 种子探索实验及报告
.\.venv\Scripts\python.exe run_real_pipeline.py
```

下载可续传，已完成的批量实验会跳过。全部数据约 780 MB，网络速度可能成为主要耗时；训练成本以本机实测为准。

命令、划分、方法与预测输入约定详见 [真实数据运行说明](REAL_DATA_GUIDE.md)。当前环境见 [版本核对](config_inventory.md) 和 `environment_verified.json`；`requirements-environment.txt` 是原始版本快照，`requirements-repro.txt` 是移除本机 wheel 路径后的重建清单；当前 pip check 已通过，独立干净环境重建尚未验证。

## 实验约束

- 每轮一个目标被试：T 会话为校准池，E 会话固定最终测试。
- 另一个源被试提供验证集，其余源被试训练；目标测试集不用于选参。
- 所有适配复制同一源模型，使用同一批嵌套校准样本；冻结主干时也冻结 BN 运行统计。
- 保存原始试次 ID、配置、数据摘要、逐试次预测、权重和成本，自动检查模型重载预测一致。
- 当前参数为探索性预设；正式选参、全部数据评估和研究结论需分别验收。

## 论文来源

Lawhern et al. (2018), EEGNet，DOI [10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c)。借鉴时间卷积、空间深度卷积、可分离卷积及紧凑设计。原论文已有跨被试实验；本项目继续研究少样本个体适配与成本。

当前真实数据模型对照作者结构实现，但采样率、时间窗、训练划分和部分超参数不同，不是原论文结果严格复现。通道仿射调制也不是可直接声称的新发明。详细对照与相关工作见 [文献记录](references/LITERATURE_NOTES.md)。

## 历史合成演示

```powershell
.\.venv\Scripts\python.exe run_experiment.py
.\.venv\Scripts\python.exe run_gpu_experiment.py
```

这些旧入口只产生合成数据结果，输出到 `results/` 顶层。旧网络为 1,020 参数的简化模型；旧“校准”是混合源样本与两个目标被试校准样本后从头训练。旧 `subject_normalized` 实际为逐试次逐通道时间标准化。原来 97.58%/100% 等数字不是实际运动想象性能。

最初的目标与任务列表保留在 [历史交接](PROJECT_HANDOFF.md) 和 [实施计划](IMPLEMENTATION_PLAN.md)，其“本轮”描述对应建档时点，最新进度以后续实验文件和进度记录为准。

最新九被试真实 EEG 探索实验的完整、可追溯记录见 [详细实验记录](DETAILED_EXPERIMENT_LOG.md)。

项目重整与论文对照方案见 [PROJECT_REDESIGN.md](PROJECT_REDESIGN.md)。当前主线已收敛为论文基线复现 + 预注册少样本适配比较；通道仿射只保留为消融。

已开始论文配置基线分支，见 [PAPER_REPRODUCTION_LOG.md](PAPER_REPRODUCTION_LOG.md)。已完成九被试论文配置矩阵；结果属于探索性分析，不是论文数值的严格复现。

当前论文式基线矩阵及被试级统计见 [PAPER_REPRODUCTION_LOG.md](PAPER_REPRODUCTION_LOG.md) 和 `results/real/paper_reproduction/`。

Euclidean Alignment 的隔离试验记录见 [EUCLIDEAN_ALIGNMENT_LOG.md](EUCLIDEAN_ALIGNMENT_LOG.md)；由于首个试验显著退化，暂不纳入完整主矩阵。

当前主线已经锁定为分类头少样本适配，决策与开发/留出检查见 [HEAD_ADAPTATION_DECISION.md](HEAD_ADAPTATION_DECISION.md)。

GitHub 上传前的文件范围、忽略项和新环境检查见 [GITHUB_RELEASE_CHECKLIST.md](GITHUB_RELEASE_CHECKLIST.md)。真实数据、模型和结果矩阵默认不进入仓库。
