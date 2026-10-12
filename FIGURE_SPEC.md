# 申请材料配图设计稿（暂不生成图片）

## 图 1：模型与适配流程

建议文件名：`figures/eegnet_adaptation_pipeline.svg`

采用横向流程图，分成三个区域：

1. **输入与预处理**：22 通道 × 500 点；4–38 Hz 滤波；125 Hz；试次开始后 2–6 秒。
2. **EEGNet 风格特征提取器**：时间卷积 `1×64, 8 filters` → 深度空间卷积 `22×1` → ELU/池化/Dropout → 深度可分离卷积 `1×16 + 1×1` → 展平分类头。
3. **跨被试适配比较**：source、head、full、scale_head、affine_head 五个分支；旁注可训练参数分别为 0、964、2,420、986、1,008。

底部统一标注：目标 E 会话固定为最终测试集；校准只来自目标 T 会话；每类预算为 1/3/5/10 条。

## 图 2：准确率与适配成本

建议文件名：`figures/few_shot_adaptation_cost_accuracy.png`

使用双面板：

- 左面板：横轴为每类校准样本数 `0, 1, 3, 5, 10`，纵轴为 Accuracy；画 source、head、full、scale_head、affine_head；误差带使用被试级 95% bootstrap CI。
- 右面板：横轴为可训练参数量，纵轴为平均适配时间；每个点标注方法名，突出“参数量少不必然意味着适配更快”。

图注建议：

> Means are computed after averaging random seeds within each target subject. Error bars show 95% bootstrap confidence intervals across target subjects. The independent clean-environment run is reported separately as a reproducibility smoke test.

正式作图时应从 `results/real/paper_reproduction/` 或 `results/real/nine_subject_exploratory/` 的 CSV 读取数据，不要手工录入数字。
