# 真实 EEG 流程

此入口独立于原有合成演示。首次实施日期 2026-10-08；运行结果和实际完成范围以 `results/real/` 下完成的 metrics.json 为准。

## 命令

在 `<project-root>` 执行：

```powershell
.\.venv\Scripts\python.exe verify_environment.py
.\.venv\Scripts\python.exe download_real_data.py
.\.venv\Scripts\python.exe prepare_real_data.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
# 先跑一次目标被试 1，源域验证被试 2，其余 7 个被试训练
.\.venv\Scripts\python.exe run_real_experiment.py --target 1 --seed 7
.\.venv\Scripts\python.exe summarize_real_results.py --root results/real/pilot
# 完整探索矩阵：9 目标 × 3 种子，各含 4 预算及缩放消融
.\.venv\Scripts\python.exe run_real_batch.py
# 或一条命令串联下载、预处理、全部探索实验与报告
.\.venv\Scripts\python.exe run_real_pipeline.py
```

下载器仅写项目 data 目录，使用 HTTPS 证书验证、上游最多 4 个、NEMAR 镜像最多 8 个并发连接、分段续传、MAT 内容验证和本地 SHA256。SHA256 用于本地复现追溯，不是发布方签名。预处理通过已安装 MOABB 读取 BNCI MAT 转换数据。支持 `--subjects 1 2 3` 部分获取；小规模运行同样必须显式传 `--subjects 1 2 3`，此时源训练仅一人，不能当作全数据跨被试结果。

上游下载过慢时使用 `download_real_data.py --provider nemar --workers 4`。本轮取得的 NEMAR nm000139 v1.0.2 manifest 在 references/；下载文件须与该清单的大小和 SHA256 一致。完整 pipeline 默认使用此镜像，独立下载脚本默认上游。

## 环境重建

本机 `pip check` 已通过，CUDA 实际计算通过。使用 Windows x64 / Python 3.14 时，可在新虚拟环境执行 `python -m pip install -r requirements-repro.txt`；该文件将原始 freeze 中本机 Torch wheel 的绝对路径替换为精确的 `torch==2.11.0+cu128` 和官方 CUDA wheel 索引。`requirements-environment.txt` 保留原始安装来源快照。当前没有额外下载数 GB 的 wheel 重建第二个完整环境，所以不能宣称干净重建已实测通过。

## 数据和协议

- 上游：https://lampx.tugraz.at/~bci/database/001-2014/ 。原始说明：[description.pdf](https://lampx.tugraz.at/~bci/database/001-2014/description.pdf)。
- 9 个被试，每人 T/E 两会话，每会话通常 288 试次，四类均衡。标签映射：0 左手、1 右手、2 双脚、3 舌头。
- 原始采样 250 Hz，22 EEG + 3 EOG；只取 22 EEG，不将 EOG 用于分类。
- MAT `trial` 是试次开始；MOABB 在该位置写入类别事件。提示在其后 2 秒。取 [2,6) 秒、4–38 Hz 零相位 FIR、重采样 125 Hz，缓存为 22×500、微伏。
- 当前默认保留微伏单位（`--input-scale 1`），不拟合目标测试统计。早期 `three_subject_pilot` 使用除以 100 的比例，导致小数据 BatchNorm 统计收敛问题；该失败试验仍完整保留。修正依据源被试运行验证，诊断入口为 diagnose_source_scale.py。按原始 run 独立滤波；本项目是离线试次分类，不是原比赛要求的连续因果分类。
- 保留所有含伪迹标记试次。原比赛部分指标基于无伪迹试次，因此本项目结果不能直接与原比赛排名数字比较。
- 目标 T 会话仅用于校准池，E 会话固定测试。目标编号之后的下一个源被试（循环）提供验证数据，其余源被试的两个会话训练。
- 每类校准 1/3/5/10 条，按种子生成嵌套索引。每个方法共享源检查点、校准集、最终测试集；无适配与 CSP 不使用校准标签。
- 源模型：Adam 1e-3，最多 100 轮，验证损失早停 patience=20。适配：Adam 1e-3，50 次全校准批次更新，无目标测试早停。这是预设探索配置，尚未完成源域模拟适配选参。
- 所有适配均固定源 BatchNorm 运行统计、关闭 Dropout；全模型微调更新原网络的全部可训练权重（包括 BN affine），不更新新增通道参数。

## 模型结构依据与差异

原论文：[Lawhern et al. 2018](https://doi.org/10.1088/1741-2552/aace8c)。作者代码：[arl-eegmodels](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py)，本地快照及原许可证在 references/。

|部分|作者默认实现|本项目真实数据实现|
|---|---|---|
|时域卷积|F1=8，长度64，SAME|相同维数，显式非对称补零|
|空间深度卷积|D=2，空间核跨全部通道，max-norm 1|groups=8，16 输出，空间核范数限制1|
|池化/激活|ELU，4倍池化，Dropout|同结构，Dropout 0.5|
|可分离卷积|长度16，F2=16|depthwise+pointwise，长度16，16输出|
|第二池化|8倍池化，Dropout，Flatten|同结构|
|分类|Dense max-norm .25 + Softmax|Linear max-norm .25；交叉熵接收 logits，预测时 Softmax|
|BN|Keras 默认 eps=.001，momentum=.99|eps=.001，PyTorch momentum=.01；更新规则仍有框架细节差异|

作者默认长度 64 针对约 128 Hz，而作者说明其 SMR 数据使用长度 32。本项目在 125 Hz 使用 64，是明确的配置选择，不能宣称严格复现 SMR 实验。初始化使用 PyTorch 默认，优化器细节、数据划分和适配流程也不同。

原网络可训练参数 2,420；新增 22 个缩放和 22 个偏移后总计 2,464。head 更新 964；scale_head 更新 986；affine_head 更新 1,008；full 更新 2,420。源训练冻结新增 44 参数。轻量适配约更新总参数的 41%，需实测收益和成本，不能仅凭名称宣称优势。

## 输出与重载

每个 target/seed 目录保存 config.json、splits.json、source.pt、各适配权重、逐试次预测、混淆矩阵、训练/适配损失和 metrics.json/csv。每次评估自动核查重新加载的预测一致。

`audit_real_results.py --root <结果目录>` 独立复核划分、嵌套与类别平衡、代码快照 SHA256、预测标签、指标、混淆矩阵及 CSP 重载。源模型只输出一个类别时会记录诊断警告，不把工程跑通等同于可用的分类性能。

```powershell
.\.venv\Scripts\python.exe predict_real.py --checkpoint results/real/pilot/target_01_seed_7/affine_head_k5.pt --data data/processed/subject_01.npz --output results/real/pilot/offline_predictions.npz
.\.venv\Scripts\python.exe evaluate_checkpoint.py --run-dir results/real/pilot/target_01_seed_7 --method affine_head --budget 5
```

预测输入必须已经按上述流程预处理；该示例对缓存中所有试次输出结果，不能把这些结果整体当作独立测试准确率。正式测试指标始终只对应 splits.json 中的 test。

报告先对每个被试的种子结果取平均，再计算被试间波动。单被试不产生有意义的被试间标准差。种子不是独立被试，未做显著性声明。

解释限制：4 Hz 高通已去掉大部分原始慢偏移，新增 offset 的作用是模型输入的可学习平移，不能直接解释为恢复/纠正电极的原始 DC 偏移。

数据使用条件：MOABB 1.7.2 元数据标注 CC-BY-ND-4.0，此为库中的声明，不等于已核实发布方对所有分发版本的许可。当前仅本地研究，不再分发数据或处理后的缓存；公开发布数据前应核实上游条件。代码和结果引用需注明数据集与 EEGNet 来源。相关适配方法文献综述尚待补齐，不宣称通道仿射变换为新方法。
