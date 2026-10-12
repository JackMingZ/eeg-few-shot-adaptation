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

审计脚本通过以下检查：划分无交集、校准预算嵌套、代码快照、输入缓存哈希、共享源检查点哈希、预测 ID 与标签、指标重算、混淆矩阵和模型重载。

审计命令：

```powershell
D:\special_project_clean_env\Scripts\python.exe `
  D:\special_project\audit_real_results.py `
  --root D:\special_project\results\clean_env_smoke\target_01_seed_7\target_01_seed_7
```

审计输出中的 `completed_runs: 0` 是因为该脚本针对批量实验根目录统计完成组合，而本次输入是单个目标目录；逐项检查均已通过，不能把这个计数解释为运行失败。

## 解释边界

本记录证明代码能够在独立环境中运行，不能证明短训练设置下的模型性能，也不能替代完整实验的统计结论。GPU 用户仍应使用 `requirements-gpu.txt` 或 PyTorch 官方 CUDA wheel 做独立 GPU 验证。
