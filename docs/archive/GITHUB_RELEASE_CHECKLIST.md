> 历史材料，保留开发过程；当前配置与结论以 [首页](../../README.md) 为准。

# GitHub 发布审查

## 建议上传

- `src/`、`tests/` 和根目录 Python 入口；
- `README.md`、`REAL_DATA_GUIDE.md`、`PROJECT_REDESIGN.md`、`HEAD_ADAPTATION_DECISION.md`；
- `PAPER_REPRODUCTION_LOG.md`、`EUCLIDEAN_ALIGNMENT_LOG.md`、`DETAILED_EXPERIMENT_LOG.md`；
- `references/` 中的文献笔记和引用链接；
- `requirements.txt`、`requirements-gpu.txt`、`requirements-repro.txt`；
- `APPLICATION_PROJECT.md`、`MODEL_PROFILE.md`；
- `LICENSE`（仅覆盖本项目代码和文档，数据集及第三方实现遵循各自条款）；
- `CITATION.cff`；
- `.github/workflows/tests.yml`；
- 小型历史说明文件。真实数据、模型检查点和批量结果不应提交。

## 不应上传

- `data/`：包含公开数据的本地缓存，体积大且有独立数据使用条件；
- `results/`：包含模型权重、逐试次预测、日志和大量生成图表；
- `.venv/`、`__pycache__/`、`.pytest_cache/`；
- `*.whl`、`*.pt`、`*.log`；
- `environment_verified.json`：包含本机路径、GPU 和操作系统信息，仅作为本地验证产物。

## 发布前检查

```powershell
git status --short --ignored
git check-ignore -v data results .venv *.whl *.pt *.log environment_verified.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m py_compile *.py
```

## 新环境运行

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify_environment.py
.\.venv\Scripts\python.exe download_real_data.py
.\.venv\Scripts\python.exe prepare_real_data.py
```

GPU 用户应根据 PyTorch 官方 CUDA 安装说明选择对应 wheel。`requirements-repro.txt` 是当前 Windows/Python 3.14 环境快照，不保证所有平台可直接安装。

`requirements.txt` 现在包含 PyTorch CPU/默认安装入口；需要 CUDA 的用户可使用 `requirements-gpu.txt` 或按 PyTorch 官方页面选择对应 CUDA wheel。

独立 CPU 环境已完成安装、`pip check`、6 项协议测试和小规模真实 EEG smoke test；记录见 `REPRODUCIBILITY_SMOKE_TEST.md`。这项 smoke test 验证运行链路，不替代完整九被试实验。

## 数据和许可证说明

代码、文档和实验脚本可以单独上传。BNCI2014-001 原始数据不随仓库分发；使用者应根据官方数据页面和 MOABB/NEMAR 元数据自行下载并遵守数据使用条件。论文、数据集和 EEGNet 作者代码分别按其原始引用和许可证处理。

## 当前项目状态

可公开表述为：真实公开 EEG 数据上的 EEGNet 风格跨被试少样本适配研究工程。当前主方法是分类头微调，全模型微调是高成本对照；论文配置矩阵和探索性结果均已保存为本地生成产物，不应在没有重新运行的情况下声称任意环境可复现相同数字。仓库公开前应确认 README、引用信息和自动测试均为最新状态。
