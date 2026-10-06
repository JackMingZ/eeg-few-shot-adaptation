# Cross-Subject EEG Calibration

一个优先使用 GPU、没有 GPU 时自动回退 CPU 的脑机接口小项目：研究被试级频域归一化和少量校准是否能改善跨被试运动想象 EEG 分类。

## 项目问题

EEGNet 类方法在训练被试上通常表现良好，但换到新被试时会受到电极阻抗、脑区差异和信号幅值变化影响。本项目构造一个可控的 EEG 数据实验，比较：

1. 全局标准化 + 频域特征 + 逻辑回归；
2. 被试级频域归一化 + 同一分类器；
3. 被试级归一化 + 新被试少量标注样本校准。

## 运行

```powershell
cd D:\special_project
python run_experiment.py
# GPU 优先的 EEGNet 版本
python run_gpu_experiment.py
```

输出位于 `results/`：

- `metrics.csv`：每个方法的 accuracy、macro-F1、kappa
- `accuracy_comparison.png`：方法对比图
- `confusion_matrix.png`：最佳方法的混淆矩阵
- `experiment_summary.json`：实验配置和结果摘要
- `gpu_metrics.json`：GPU/CPU EEGNet 训练结果和参数量

## 研究解释

这不是声称提出了新的 EEGNet 网络，而是一个围绕 EEGNet 输入信号和跨被试问题的可复现实验。合适的研究结论是：在固定数据生成机制和评估协议下，被试级频域归一化与少量校准对跨被试分类是否有效，以及它在哪些类别上有效。

## GPU 配置

当前主机检测到 NVIDIA GeForce RTX 5070、595.97 驱动和 12 GB 显存。安装 `requirements-gpu.txt` 后，`run_gpu_experiment.py` 会自动使用 CUDA；如果 CUDA 不可用，会自动使用 CPU。

## 接入真实数据

后续可以将 `src/eeg_project.py` 中的 `make_synthetic_eeg` 替换为 BCI Competition IV 2a 或 PhysioNet EEG Motor Movement/Imagery 数据加载器。分类器、指标和绘图代码无需改变。
