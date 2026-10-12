# 独立干净环境复现记录

## 目的

本次运行验证项目能否在全新 Python 环境中安装依赖、通过协议测试，并完成一个小规模真实 EEG 训练流程。它是软件复现 smoke test，不是新增的确证性科学实验。

## 环境

- Python 3.14.4
- Windows 11
- 独立环境：`D:\special_project_clean_env`
- PyTorch 2.14.1+cpu
- NumPy 2.5.3
- SciPy 1.18.1
- scikit-learn 1.9.1
- pandas 3.0.6
- matplotlib 3.11.2
- MNE 1.13.2
- MOABB 1.7.2
- `pip check`：通过
- CUDA：本次环境使用 CPU PyTorch，项目 CPU 回退路径正常

## 执行的检查

```powershell
python -m venv D:\special_project_clean_env
D:\special_project_clean_env\Scripts\python.exe -m pip install -r D:\special_project\requirements.txt
D:\special_project_clean_env\Scripts\python.exe D:\special_project\verify_environment.py
D:\special_project_clean_env\Scripts\python.exe -m pip check
D:\special_project_clean_env\Scripts\python.exe -m unittest discover -s D:\special_project\tests -v
```

结果：6 项测试全部通过，依赖没有冲突。

## 小规模真实 EEG 运行

使用已经准备好的 `data/processed/` 缓存，选取被试 1、2、3，以被试 1 为目标，被试 2 为验证，被试 3 为源训练；训练 5 个 epoch，早停 patience 为 2；目标校准预算为每类 1 和 3 条。

```powershell
D:\special_project_clean_env\Scripts\python.exe `
  D:\special_project\run_real_experiment.py `
  --data-root D:\special_project\data\processed `
  --subjects 1 2 3 --target 1 --seed 7 `
  --epochs 5 --patience 2 --adapt-epochs 5 `
  --budgets 1 3 --skip-csp `
  --output D:\special_project\results\clean_env_smoke\target_01_seed_7
```

输出目录为：

`results/clean_env_smoke/target_01_seed_7/target_01_seed_7/`

运行生成了配置、源模型、5 种方法的预测文件、适配模型和指标文件。源模型训练耗时约 1.78 秒；由于这是缩短训练的 smoke test，目标测试准确率约为 24.0%–25.3%，不能与完整九被试实验的准确率比较。

## 复现检查结果

审计脚本通过以下检查：划分无交集、校准预算嵌套、代码快照、输入缓存哈希、共享源检查点哈希、预测 ID 与标签、指标重算、混淆矩阵。神经网络预测重载一致性由训练脚本检查，本次未运行 CSP。

审计命令：

```powershell
D:\special_project_clean_env\Scripts\python.exe `
  D:\special_project\audit_real_results.py `
  --root D:\special_project\results\clean_env_smoke\target_01_seed_7
```

更正（2026-10-12）：之前传入单个目标目录，`completed_runs: 0` 表示没有审计任何实验，原来的“逐项检查均已通过”表述错误。本次使用上面的父目录重新执行，实际审计 1 个运行并通过。代码现在对空输入报错，防止重复误报。

## 解释边界

本记录证明代码能够在独立环境中运行，不能证明短训练设置下的模型性能，也不能替代完整实验的统计结论。GPU 用户仍应使用 `requirements-gpu.txt` 或 PyTorch 官方 CUDA wheel 做独立 GPU 验证。


## 主配置的新流程检查（2026-10-12）

同一独立 CPU 环境又运行了 128 Hz / 256 点配置，3 名被试、目标 1、种子 7、训练 5 轮、适配 5 轮、预算 1/3。输出 `results/real/public_workflow_smoke`，包含 source、4 种适配各两个预算及 CSP，共 10 行。
批量入口、无图片汇总及审计均完成，审计实际读取 1 个运行，包括 CSP 重载检查；没有生成图片。此次复用已有预处理缓存，没有重测下载/预处理或完整矩阵。

```powershell
python run_real_batch.py --subjects 1 2 3 --targets 1 --seeds 7 --epochs 5 --adapt-epochs 5 --budgets 1 3 --data-root data/processed_paper --samples 256 --sampling-rate 128 --epoch-start 2.5 --epoch-end 4.5 --temporal-kernel 32 --dropout 0.25 --output results/real/public_workflow_smoke --no-plots
```

协议测试现为 9 项，新增空审计拒绝、配置一致性和主配置适配参数检查。旧记录的 6 项是当时实际数量。
