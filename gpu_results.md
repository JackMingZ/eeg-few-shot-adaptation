# GPU 实验结果

运行命令：

```powershell
& "D:\\special_project\\.venv\\Scripts\\python.exe" run_gpu_experiment.py
```

实际使用设备：NVIDIA GeForce RTX 5070，CUDA 12.8，PyTorch 2.11.0。

| 方法 | Accuracy | Macro-F1 | Kappa | 参数量 |
|---|---:|---:|---:|---:|
| EEGNet 风格网络，仅使用训练被试 | 0.9758 | 0.9757 | 0.9677 | 1,020 |
| EEGNet 风格网络 + 校准样本混合后从头训练 | 1.0000 | 1.0000 | 1.0000 | 1,020 |

结果保存在 `results/gpu_metrics.json` 和 `results/gpu_accuracy_comparison.png`。

当前数据是固定随机种子的可控 EEG 合成数据，用于验证 GPU 训练链路和实验设计。接入真实公开 EEG 数据后，应重新运行并报告真实数据结果。
