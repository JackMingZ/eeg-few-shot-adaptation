# 论文与方法核对（2026-10-08）

## 已核对：EEGNet

Lawhern et al., EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–Computer Interfaces, Journal of Neural Engineering (2018), DOI [10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c)。本轮实际读取的是 [arXiv v4 全文](https://arxiv.org/abs/1611.08024v4)（标题使用 “Convolutional Network”），并对照作者 [EEGModels.py](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py)。保留 PDF、文本抽取和代码快照。

灵感对应全文 2.2.1 节及表 2：时间卷积学习频率滤波器，空间深度卷积学习各频率对应的空间滤波器，可分离卷积解耦时域概括与特征图组合，以紧凑参数量分类。原型的频带功率逻辑回归不等于 EEGNet 的网络实现。

需要补充历史聊天的表述：**原论文本身已经开展跨被试实验**（2.3 节），并非本项目首次提出“换被试评估”。本项目选择的后续问题是小样本个体适配及成本比较。

|设置|论文 SMR 实验|本项目探索设置|
|---|---|---|
|数据|BCI IV 2a|同数据集的 BNCI MAT 版本|
|采样率|128 Hz|125 Hz|
|窗口|提示后 0.5–2.5 秒|提示后 0–4 秒|
|时间核|32|64（作者实现的默认值）|
|跨被试 Dropout|0.25|0.5（作者实现的默认值）|
|源训练/验证|随机 5 人训练，3 人验证；只用其训练会话|循环固定 1 人验证，其余源被试训练；使用源被试两个会话|
|目标测试|原比赛评估会话|同样使用目标 E 会话，但保留伪迹试次|
|重复|每目标 10 次，共 90 folds|计划 3 种子；实际完成数由指标文件确定|
|训练|500 轮并按验证损失保存|最多 100 轮、patience 20 早停|
|目标适配|该跨被试协议不使用目标训练会话|从目标 T 会话取少量有标签试次适配|

因此，本实现可用作 EEGNet 结构基线，但不是论文实验严格复现；参数量和准确率不能直接对标论文表格。

## 已核对摘要：相关一般方法

- Perez et al., [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871v2)。摘要明确提出 feature-wise affine transformation。通道缩放与偏移属于已有仿射调制思想；本项目是在 EEG 输入端通过目标标签直接优化参数，不等同于 FiLM 的条件生成机制，也不能声称发明了仿射调制。
- Li et al., [Revisiting Batch Normalization For Practical Domain Adaptation](https://arxiv.org/abs/1603.04779v4)。摘要提出通过改变各 BN 层统计量进行 AdaBN 适配。当前项目固定 BN 统计，方法与 AdaBN 不同；未来若纳入此对照，必须明确统计量仅使用哪个目标数据池。

以上两篇本轮只核对 arXiv API 标题与摘要，不冒充阅读全文或全面综述。

补充检索并核对摘要的 EEG 相关工作：

- [A Systematic Evaluation of Euclidean Alignment with Deep Learning for EEG Decoding](https://arxiv.org/abs/2401.10746v4)：研究 EA 与多被试深度学习、向新被试迁移。与本项目相关，但 EA 是协方差对齐，不是逐通道可学习缩放/偏移。
- [Revisiting Euclidean Alignment for Transfer Learning in EEG-Based Brain-Computer Interfaces](https://arxiv.org/abs/2502.09203v2)：讨论 EA 的正确使用及扩展，后续加入对齐基线前需要阅读全文，尤其核实目标统计量的使用范围。
- [Graph Adapter of EEG Foundation Models for Parameter Efficient Fine Tuning](https://arxiv.org/abs/2411.16155v2)：冻结主干、训练图适配器，说明 EEG 参数高效适配已有相关研究；其基础模型、任务与本项目运动想象设置不同，摘要中的性能数值不可移用于本项目。

搜索结果快照见 eeg_alignment_search.xml 和 eeg_affine_search.xml。此为初步定位，不是系统综述，未确认有无与本项目完全相同的方案，不能据此声称新颖性。

## 数据说明核对

[官方数据说明](https://lampx.tugraz.at/~bci/database/001-2014/description.pdf)已读取：22 EEG、3 EOG、250 Hz、9 被试、两会话、6 runs/会话、48 试次/run、提示在试次开始后 2 秒出现，6 秒结束。EOG 不应用于分类。原比赛是连续因果输出协议，最终计分的伪迹政策与当前离线试次评估不同，不应混称。

NEMAR nm000139 v1.0.2 的 manifest 和 sourcedata provenance 记录原始 MAT 文件大小及 SHA256，可核对 BNCI 与镜像文件一致。NEMAR 元数据标注 CC-BY-ND-4.0，来源 DOI 为 10.82901/nemar.nm000139；本项目不重新分发数据。
