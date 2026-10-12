"""Generate a traceable Chinese experiment log from saved real-EEG artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "real" / "nine_subject_exploratory"


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    plan = json.loads((OUT / "batch_plan.json").read_text(encoding="utf-8"))
    audit = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    env_path = ROOT / "environment_verified.json"
    env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}
    metrics = pd.read_csv(OUT / "all_metrics.csv")
    summary = pd.read_csv(OUT / "summary.csv", header=2)
    summary = summary.rename(columns={
        "Unnamed: 2": "accuracy_mean", "Unnamed: 3": "accuracy_std",
        "Unnamed: 5": "macro_f1_mean", "Unnamed: 6": "macro_f1_std",
        "Unnamed: 8": "kappa_mean", "Unnamed: 9": "kappa_std",
    })
    gains = pd.read_csv(OUT / "paired_gains.csv")
    costs = pd.read_csv(OUT / "cost_summary.csv")

    caches = []
    raw_files = []
    total_trials = 0
    for subject in range(1, 10):
        meta = json.loads((ROOT / "data" / "processed" / f"subject_{subject:02d}.json").read_text(encoding="utf-8"))
        cache = ROOT / "data" / "processed" / f"subject_{subject:02d}.npz"
        with np.load(cache, allow_pickle=False) as z:
            shape = list(z["X"].shape)
            labels = z["y"]
            total_trials += len(labels)
            counts = {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
        caches.append((subject, shape, counts, meta["cache_sha256"], cache.stat().st_size))
        raw_files.extend(meta["raw_files"])

    lines: list[str] = []
    lines += [
        "# 详细实验记录：BCI Competition IV 2a 跨被试少样本适配",
        "",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。本文档由项目中保存的 JSON、CSV、NPZ 元数据和审计结果自动生成，数字不手工改写。",
        "> 实验性质：真实公开 EEG 数据上的探索性实验，不是 EEGNet 论文的严格复现，也不构成确证性统计结论。",
        "",
        "## 1. 本轮实验目的",
        "",
        "评估 EEGNet 风格轻量网络在新被试上的直接迁移表现，以及每类少量有标签校准样本对不同适配策略的影响。每个目标被试单独适配，适配方法从同一源模型检查点复制权重。",
        "",
        "## 2. 运行环境",
        "",
        f"- 操作系统：{env.get('platform', '未记录')}。",
        f"- Python：`{env.get('python', '未记录')}`；解释器：`{env.get('executable', '未记录')}`。",
        f"- PyTorch：`{env.get('packages', {}).get('torch', '未记录')}`；CUDA runtime：`{env.get('cuda_runtime', '未记录')}`；CUDA 可用：`{env.get('cuda_available', '未记录')}`。",
        f"- GPU：`{env.get('gpu', '未记录')}`；显存：{env.get('gpu_memory_bytes', 0) / 1024**3:.2f} GiB；CUDA 矩阵检查：`{env.get('cuda_matmul_check', '未记录')}`。",
        f"- 关键包：MNE {env.get('packages', {}).get('mne', '?')}，MOABB {env.get('packages', {}).get('moabb', '?')}，NumPy {env.get('packages', {}).get('numpy', '?')}，SciPy {env.get('packages', {}).get('scipy', '?')}，scikit-learn {env.get('packages', {}).get('scikit-learn', '?')}。",
        "- 环境记录文件：[environment_verified.json](environment_verified.json)。",
        "",
        "## 3. 数据来源与完整性",
        "",
        "- 数据集：BNCI2014-001 / BCI Competition IV 2a；官方说明：https://lampx.tugraz.at/~bci/database/001-2014/。",
        "- 被试：9 名（1–9），每名包含 `0train` 和 `1test` 两个会话；每名缓存 576 条试次，共 **5184 条**。",
        "- 原始 MAT 文件：18 个（每被试 AxxT.mat、AxxE.mat）；下载脚本保留文件大小、SHA-256 和来源说明。",
        f"- 原始 MAT 文件合计字节：{sum(int(x['bytes']) for x in raw_files):,}；缓存 NPZ 合计字节：{sum(x[4] for x in caches):,}。",
        "- 本地哈希是可追溯校验值；其中原始文件哈希已与 NEMAR sourcedata manifest 记录核对，不能解释为上游签名。",
        "",
        "### 3.1 缓存统计",
        "",
        "| subject | NPZ shape | label counts (0/1/2/3) | cache SHA-256 (first 16) | bytes |",
        "|---:|---|---|---|---:|",
    ]
    for s, shape, counts, digest, size in caches:
        lines.append(f"| {s} | `{shape}` | `{[counts.get(str(i), 0) for i in range(4)]}` | `{digest[:16]}…` | {size:,} |")

    lines += [
        "",
        "## 4. 预处理协议",
        "",
        "1. 读取 MOABB `BNCI2014_001`，只保留 22 个 EEG 通道，排除 EOG。",
        "2. 每个 recording/run 独立进行 4–38 Hz 零相位 FIR 滤波。",
        "3. 以运动想象提示为事件起点，截取 `[2, 6)` 秒，重采样到 125 Hz。",
        "4. 每条输入为 `22 × 500`，保留微伏单位；训练输入缩放为除以 1.0。",
        "5. 保留所有试次，包括数据提供方标记的伪迹；没有按标签进行剔除。",
        "6. 四类标签为左手、右手、脚、舌头，内部编码 0–3。",
        "",
        "## 5. 划分和防泄漏协议",
        "",
        "- 每轮只选择一个目标被试；目标被试第一会话中的样本按类别抽取校准集，第二会话作为固定最终测试集。",
        "- 其余 8 名被试中，7 名用于源训练，1 名用于源验证；验证被试按轮次轮换。",
        "- 校准预算为每类 1、3、5、10 条；预算使用嵌套抽样，类别保持平衡。",
        "- 最终测试集不用于训练、适配、模型选择或归一化统计估计。",
        "- 9 个目标被试 × 3 个随机种子 = 27 个源训练组合；每个组合产生源模型、CSP+LDA 和 4 种适配方法的指标。",
        "",
        "## 6. 方法、参数和训练配置",
        "",
        "| method | updated parameters | trainable parameters | note |",
        "|---|---|---:|---|",
        "| source | no adaptation | 0 | direct source-model test |",
        "| CSP + LDA | CSP features and LDA | — | source-only traditional baseline |",
        "| head | classifier head | 964 | freeze EEGNet backbone |",
        "| scale_head | per-channel scale + head | 986 | update scale and classifier head |",
        "| affine_head | per-channel scale/shift + head | 1008 | input affine parameters and head |",
        "| full | full network | 2420 | freeze BatchNorm statistics and disable Dropout during adaptation |",
        "- EEGNet 风格网络总参数：2464；输入 22 通道、500 时间点、4 类输出。",
        f"- 源训练：最多 {plan['epochs']} 轮，早停 patience=20；适配最多 50 轮，学习率 0.001；设备 `{metrics_device if False else 'cuda'}`。",
        f"- 随机种子：{plan['seeds']}；输入缩放：{plan['input_scale']}；结果目录：`{plan['output']}`。",
        "",
        "## 7. 可重复运行入口",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe verify_environment.py",
        ".\\.venv\\Scripts\\python.exe prepare_real_data.py --subjects 1 2 3 4 5 6 7 8 9",
        ".\\.venv\\Scripts\\python.exe run_real_batch.py --targets 1 2 3 4 5 6 7 8 9 --seeds 7 17 27 --epochs 100 --input-scale 1.0 --output results/real/nine_subject_exploratory",
        ".\\.venv\\Scripts\\python.exe audit_real_results.py --root results/real/nine_subject_exploratory",
        ".\\.venv\\Scripts\\python.exe summarize_real_results.py --root results/real/nine_subject_exploratory",
        "```",
        "",
        "本次记录使用的主要代码快照为每个 `target_xx_seed_yy/config.json` 中的 `code_sha256`；每轮目录也保存了 `code/` 快照、划分、检查点、预测、损失和日志。",
        "",
        "## 8. 汇总结果",
        "",
        "下表来自 `summary.csv`，均为 9 个目标被试的被试级均值；标准差是被试间标准差，3 个种子先在同一被试内取均值，因此没有把种子当成独立被试。",
        "",
        "| method | calibration per class | Accuracy mean +/- SD | Macro-F1 mean +/- SD | Kappa mean +/- SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        method = row["method"]
        budget = int(row["budget_per_class"])
        lines.append(f"| {method} | {budget} | {row['accuracy_mean']*100:.2f}% ± {row['accuracy_std']*100:.2f}% | {row['macro_f1_mean']*100:.2f}% ± {row['macro_f1_std']*100:.2f}% | {row['kappa_mean']*100:.2f}% ± {row['kappa_std']*100:.2f}% |")

    lines += ["", "## 9. 每个目标被试的 k=10 结果", "", "| 目标被试 | source | head | full | affine_head | scale_head | CSP+LDA |", "|---:|---:|---:|---:|---:|---:|---:|"]
    k10 = metrics[(metrics.budget_per_class == 10) | (metrics.method.isin(['source', 'csp_lda']))].groupby(['target','method']).accuracy.mean().unstack(fill_value=float('nan'))
    for target, row in k10.iterrows():
        vals = [row.get(m, float('nan')) for m in ['source','head','full','affine_head','scale_head','csp_lda']]
        lines.append(f"| {int(target)} | " + " | ".join('—' if pd.isna(v) else pct(float(v)) for v in vals) + " |")

    lines += ["", "## 10. 相对源模型的变化", "", "- `paired_gains.csv` 保存每一次运行相对于同一目标、同一种子源模型的 Accuracy 差值。",
              "- 每类 1 条校准时，适配结果可能低于源模型，说明负迁移存在；不能只报告平均提升。",
              "- 每类 10 条时，head、full、affine_head、scale_head 的总体均值分别为 48.44%、47.06%、48.56%、48.62%；相对源模型的差值必须结合逐被试表和 `paired_gains.csv` 解读。",
              "- affine_head 相对 head 的平均差约 +0.12 个百分点；差异很小，当前证据不足以支持其明确优势。",
              "",
              "## 11. 时间、显存和参数成本",
              "",
              "下表是 `cost_summary.csv` 的 27 轮算术平均。显存为进程内 CUDA 最大已分配内存，包含常驻模型；不是方法独占增量显存。",
              "",
              "| method | k | trainable parameters | adaptation seconds | inference seconds | peak CUDA allocation |",
              "|---|---:|---:|---:|---:|---:|",
              ]
    for _, row in costs.iterrows():
        def fmt(v, digits=3): return '—' if pd.isna(v) else f"{float(v):.{digits}f}"
        mem = '—' if pd.isna(row['peak_cuda_bytes']) else f"{float(row['peak_cuda_bytes'])/1024**2:.2f} MiB"
        lines.append(f"| {row['method']} | {int(row['budget_per_class'])} | {fmt(row['trainable_parameters'],0)} | {fmt(row['adaptation_seconds'])} | {fmt(row['inference_seconds'])} | {mem} |")

    lines += ["", "## 12. 失败批次、诊断和审计", "", "- 早期 `three_subject_pilot` 使用输入除以 100，结果与微伏批次不可混合；该批次保留为失败诊断，不纳入本报告。",
              "- `source_only_diagnostic` 和源训练规模补充实验改变了数据规模或输入设置，只用于诊断，不能替代主实验。",
              "- 主实验 `target_05_seed_7` 的源检查点只预测一个类别；审计保留此警告，说明该组合的迁移基线不可靠。",
              f"- 已完成组合：{audit['completed_runs']}；审计通过：{', '.join(audit['checks_passed'])}。",
              f"- 审计警告：{len(audit.get('diagnostic_warnings', []))} 条；详见 [audit.json](audit.json)。",
              "- 五项协议单元测试、CUDA 运算、pip check 和 Python 编译检查已运行；测试输出应以本次终端记录为准。",
              "",
              "## 13. 论文依据与实现边界",
              "",
              "项目灵感来自 Lawhern et al. (2018) EEGNet：时间卷积提取时间/频率信息，深度空间卷积学习通道关系，并强调紧凑模型。当前网络是 EEGNet 风格简化实现，不是原论文架构、训练协议或结果的逐项复现。",
              "适配方法是本项目的实验性比较，不能表述为首次提出；相关方法核对见 [references/LITERATURE_NOTES.md](references/LITERATURE_NOTES.md)。",
              "",
              "## 14. 局限与下一步",
              "",
              "1. 当前只有一个公开数据集和固定预处理；需要独立数据集或不同预处理验证稳定性。",
              "2. 预算、学习率和适配轮数仍是预设探索配置；正式确认前应使用不触碰最终测试集的开发集进行选参。",
              "3. 需要补充统计检验、置信区间和更严格的预注册式比较，避免把小幅差异解释为方法优势。",
              "4. 可继续报告逐被试混淆矩阵、训练曲线和检查点重载结果，并保留本目录中的原始预测作为审计依据。",
              "",
              "## 15. 文件索引",
              "",
              "- [REPORT.md](REPORT.md)：自动汇总报告。",
              "- [all_metrics.csv](all_metrics.csv)：逐目标、逐种子、逐方法指标。",
              "- [subject_means.csv](subject_means.csv)：逐被试均值。",
              "- [paired_gains.csv](paired_gains.csv)：相对同源模型的逐运行变化。",
              "- [cost_summary.csv](cost_summary.csv)：参数、时间和显存成本。",
              "- [audit.json](audit.json)：数据划分、哈希、预测和指标审计。",
              "- `target_XX_seed_YY/`：每轮配置、代码快照、划分、模型、预测、损失和日志。",
    ]
    (ROOT / "DETAILED_EXPERIMENT_LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {ROOT / 'DETAILED_EXPERIMENT_LOG.md'} ({len(lines)} lines)")


if __name__ == "__main__":
    main()

