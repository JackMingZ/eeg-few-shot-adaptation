# special_project 配置清单

生成时间：2026-10-06

## 工作目录

- 路径：`D:\special_project`
- 初始状态：空目录
- 当前用途：脑机接口小型研究项目
- 文件系统权限：当前用户可读写；目录所有者为 `DESKTOP-U7D2AM1\Jack_Ming`

## 主机与开发工具

- 操作系统：Windows
- Python：3.14.4
- Python 可执行文件：`C:\Program Files\Python314\python.exe`
- pip：`C:\Program Files\Python314\Scripts\pip.exe`
- Git：`D:\杂七杂八的项目\Git\cmd\git.exe`
- GPU：本项目不依赖 GPU

## 已安装科学计算依赖

- numpy 2.5.3
- scipy 1.18.1
- scikit-learn 1.9.1
- pandas 3.0.6
- matplotlib 3.11.2

## GPU 检测

- GPU：NVIDIA GeForce RTX 5070
- 驱动：595.97
- 显存：12227 MiB
- CUDA 训练：已提供 PyTorch EEGNet 路径；运行时自动检测 `cuda`

## 未预装依赖

- PyTorch
- MNE-Python
- MOABB
- Jupyter

## 研究约束与决策

1. 当前没有实验采集设备，因此使用可复现的公开数据接口和 EEG 生成器。
2. 当前环境没有深度学习框架，因此第一版使用 EEGNet 论文中的“频域特征 + 分类评估”思想，构建可运行的轻量基线。
3. 研究重点放在跨被试泛化和少量校准，避免把贡献写成没有证据支持的“新网络”。
4. 后续安装 MNE 或 PyTorch 后，可将数据生成器替换为真实数据加载器，保留评估流程。
