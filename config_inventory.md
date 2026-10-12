# 当前环境核对

核对日期：2026-10-08。使用项目 `.venv/Scripts/python.exe` 实测。

| 项目 | 当前值 |
|---|---|
| 系统 | Windows |
| Python | 3.14.4 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.6 |
| matplotlib | 3.11.2 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime | 12.8 |
| CUDA 可用 | 是 |
| GPU | NVIDIA GeForce RTX 5070 |
| MNE | 1.13.2 |
| MOABB | 1.7.2 |

2026-10-08 已用 verify_environment.py 重新核查：驱动 595.97、显存 12227 MiB，CUDA 矩阵乘法通过。机器可读记录见 environment_verified.json。requirements-environment.txt 保存全部已安装包版本，原 requirements 文件仍是最低版本约束；后续已完成独立 CPU 环境的依赖安装与小规模检查，见 REPRODUCIBILITY_SMOKE_TEST.md；完整 GPU 矩阵尚未在干净环境重跑。

本轮已新增真实数据脚本，运行说明见 REAL_DATA_GUIDE.md。最初的研究方案和待办见 PROJECT_HANDOFF.md；最新实际进度以真实实验的完成记录为准。
