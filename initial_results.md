# 初始实验记录

运行命令：`python run_experiment.py`

数据：固定随机种子的可控 EEG 合成数据；训练被试为 0–3，测试被试为 4–5。

| 方法 | Accuracy | Macro-F1 | Kappa |
|---|---:|---:|---:|
| global_baseline | 0.9315 | 0.9331 | 0.9086 |
| subject_normalized | 0.7782 | 0.7732 | 0.7043 |
| few_shot_calibration | 0.9556 | 0.9548 | 0.9409 |

当前结果说明：单独做被试级归一化会损失部分可分信息，但在加入每类少量校准样本后，模型超过全局基线。这个结果适合进一步研究“归一化与校准的组合”，不能直接外推到真实 EEG。

GPU 版本：`python run_gpu_experiment.py`。PyTorch 安装完成后将记录 CUDA/CPU 设备、参数量和 EEGNet 结果。
