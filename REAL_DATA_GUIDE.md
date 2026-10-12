# 真实 EEG 运行指南

本文所有主流程命令使用 `paper` 配置，与 [README](README.md) 和 [公开结果](reports/paper/README.md) 一致。它是论文启发的探索配置，并非严格复现原论文。

## 安装环境

以下命令在克隆后的仓库根目录执行，使用 Windows PowerShell、Python 3.14：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_environment.py
```

独立 CPU 环境已完成安装、协议测试，以及 paper/legacy 小规模训练与审计。复用过已有预处理缓存；没有在该环境重新下载全部数据或重跑完整矩阵，见 [复现记录](REPRODUCIBILITY_SMOKE_TEST.md)。

原完整实验使用 RTX 5070、PyTorch 2.11.0+cu128。GPU 用户可按 PyTorch 官方说明选取 CUDA wheel，或安装 `requirements-gpu.txt`；GPU 干净重建尚未验证。`requirements-repro.txt` 是历史 GPU 环境的精确版本清单，不能代替跨平台验证。`requirements.txt` 使用最低版本约束，不保证未来安装得到相同版本。

## 先检查一条小流程

```powershell
.\.venv\Scripts\python.exe run_real_pipeline.py --profile paper --subjects 1 2 3 --targets 1 --seeds 7 --epochs 5 --adapt-epochs 5 --budgets 1 3 --output results/real/paper_smoke --no-plots
```

入口依次下载、预处理、训练、汇总和审计。此时被试 3 训练、被试 2 验证、被试 1 的 T 会话校准、E 会话测试。短训练只检查流程，不能据此估计完整模型的性能。加 `--dry-run` 可只查看实际命令。

`--output` 接受批量结果根目录，程序自行追加 `target_01_seed_7`，不要手动再加一层。`--no-plots` 不生成图片，但仍保存指标表、权重和预测。

## 完整主配置矩阵

```powershell
.\.venv\Scripts\python.exe run_real_pipeline.py --profile paper --output results/real/paper_rerun --no-plots
```

默认九名目标被试、种子 7/17/27、预算 1/3/5/10，共 27 个运行、486 行指标。完整原始数据约 780 MB，需要网络，CPU 完整训练可能较慢。输出目录与历史 `paper_reproduction` 分开；恢复运行时会检查已完成组合的配置，配置变化请使用新目录。

如需逐步运行，以下命令与完整主配置相同（不是底层脚本的默认配置）：

```powershell
.\.venv\Scripts\python.exe download_real_data.py --subjects 1 2 3 4 5 6 7 8 9 --provider nemar --workers 4
.\.venv\Scripts\python.exe prepare_real_data.py --subjects 1 2 3 4 5 6 7 8 9 --output-dir processed_paper --sampling-rate 128 --epoch-start 2.5 --epoch-end 4.5
.\.venv\Scripts\python.exe run_real_batch.py --subjects 1 2 3 4 5 6 7 8 9 --targets 1 2 3 4 5 6 7 8 9 --seeds 7 17 27 --data-root data/processed_paper --samples 256 --sampling-rate 128 --epoch-start 2.5 --epoch-end 4.5 --temporal-kernel 32 --dropout 0.25 --epochs 100 --patience 20 --adapt-epochs 50 --adapt-lr 0.001 --input-scale 1 --budgets 1 3 5 10 --output results/real/paper_rerun --no-plots
```

只运行一个目标/种子时：

```powershell
.\.venv\Scripts\python.exe run_real_experiment.py --subjects 1 2 3 4 5 6 7 8 9 --target 1 --seed 7 --data-root data/processed_paper --samples 256 --sampling-rate 128 --epoch-start 2.5 --epoch-end 4.5 --temporal-kernel 32 --dropout 0.25 --epochs 100 --patience 20 --adapt-epochs 50 --adapt-lr 0.001 --input-scale 1 --budgets 1 3 5 10 --output results/real/paper_single
.\.venv\Scripts\python.exe summarize_real_results.py --root results/real/paper_single --no-plots
.\.venv\Scripts\python.exe audit_real_results.py --root results/real/paper_single
```

## 数据与实验协议

- BNCI2014-001 / BCI Competition IV 2a，9 人、每人 T/E 两会话，各 288 条四分类试次。标签为左手、右手、双脚、舌头（0–3）。原始说明见 [官方页面](https://lampx.tugraz.at/~bci/database/001-2014/description.pdf)。
- 保留 22 EEG 通道，排除 EOG。按原始 run 独立做 4–38 Hz 零相位 FIR；截取试次开始后 [2.5,4.5) 秒，即提示后 [0.5,2.5) 秒；重采样到 128 Hz，缓存为 22×256、微伏。保留伪迹试次。
- 目标 T 会话为校准池，E 会话固定测试。循环下一名源被试验证，其余七名的两个会话训练。预算按类别嵌套抽样，方法共享源模型和校准样本。
- 源训练 Adam 0.001，最多 100 轮，验证损失早停 patience=20；适配 Adam 0.001，50 次更新。固定 BN 运行统计，关闭 Dropout；full 更新原网络权重但不更新输入缩放/偏移。无适配与 CSP 不使用校准标签。
- 单次训练不按目标测试损失早停，但整个研究已看过测试结果并做过探索选择，不构成预注册或独立确认性验证。

时间核 32、Dropout 0.25，总参数 1,760；源训练/full 更新 1,716，head 516，scale_head 538，affine_head 560。细节见 [模型配置](MODEL_PROFILE.md)。

## 输出与核对

每个 target/seed 目录保存配置、划分、代码快照、权重、预测、损失和指标。训练时检查神经网络重载预测一致性；审计独立核对划分、嵌套预算、哈希、标签、重算指标、混淆矩阵和 CSP 重载。审计根目录必须含 `target_*/metrics.json`，空输入报错。

```powershell
.\.venv\Scripts\python.exe audit_real_results.py --root results/real/paper_rerun
```

`predict_real.py` 和 `evaluate_checkpoint.py` 仍是 legacy 专用工具，内部使用 500 点模型；不要直接用它们重载 paper 权重。主配置重载检查由训练流程执行。

公开表和统计的重算方法见 [reports/paper](reports/paper/README.md)。其中的导出器只接受固定九被试×三种子主矩阵和对应训练配置，不接受 smoke test。汇总时先平均同一被试的种子，再跨被试统计，不能把种子视为独立被试。

## 历史配置（legacy）

早期探索使用 125 Hz、试次开始后 [2,6) 秒、500 点、时间核 64、Dropout 0.5，总参数 2,464。其结果不能与首页主配置混合：

```powershell
.\.venv\Scripts\python.exe run_real_pipeline.py --profile legacy --output results/real/legacy_rerun --no-plots
```

底层 `prepare_real_data.py`、`run_real_experiment.py` 和 `run_real_batch.py` 保留 legacy 默认值；因此主配置命令必须使用 pipeline 或显式传入上述完整参数。

## 来源与范围

下载默认 NEMAR 镜像，可按需使用上游；原始文件与清单大小/SHA256 对照，哈希不是发布方签名。原始 EEG 和处理缓存不随仓库分发，使用者须遵守数据来源条款。

本项目借鉴 [EEGNet](https://doi.org/10.1088/1741-2552/aace8c) 的时空卷积结构，训练划分、轮数、伪迹政策与论文不同；这是离线试次分类，不是原比赛连续因果评分。测试通过只说明软件检查通过，不等于模型有效或具备实际部署能力。
